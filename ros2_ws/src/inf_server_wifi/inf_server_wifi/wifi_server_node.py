"""inf_server_wifi: DR-SPAAM + Kalman tracking over plain ROS 2 DDS.
"""
import math
import os
import statistics
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster

from dr_spaam.detector import Detector

from inf_server_wifi.kalman_tracker import MultiObjectTracker


RELIABILITY_POLICIES = {
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
    "reliable": ReliabilityPolicy.RELIABLE,
}

# Discovery is set per process, before rclpy starts. Logged, never written.
DISCOVERY_ENV = (
    "ROS_DOMAIN_ID",
    "ROS_LOCALHOST_ONLY",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "ROS_STATIC_PEERS",
    "RMW_IMPLEMENTATION",
)


def build_qos(reliability, depth):
    """QoS for one endpoint. Mirrors inf_client_wifi.build_qos exactly.

    The two sides must agree or DDS silently refuses the match: a RELIABLE
    subscriber receives nothing at all from a BEST_EFFORT publisher, with no
    error anywhere -- just a topic that never fires. Since that failure looks
    identical to "the robot is not running", the shape is kept in one obvious
    place on each side and tested.

    Raises ValueError on an unknown reliability so a typo in params.yaml fails
    at startup rather than silently selecting a default that changes the
    experiment.
    """
    if reliability not in RELIABILITY_POLICIES:
        raise ValueError(
            f"unknown reliability {reliability!r}; "
            f"expected one of {sorted(RELIABILITY_POLICIES)}"
        )
    return QoSProfile(
        reliability=RELIABILITY_POLICIES[reliability],
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        durability=DurabilityPolicy.VOLATILE,
    )


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class ArrivalStats:
    """Loss estimate, reorder count and jitter for the one scan stream.

    Stands in for inf_server_udp's SeqTracker, which cannot exist here: DDS
    hands over a sample with no sequence number, so the only ordering
    information available is the header timestamp the robot wrote.
    """

    
    JITTER_GAIN = 1.0 / 16.0
    PERIOD_WINDOW = 64

    
    PERIOD_MIN_SAMPLES = 16
    MISS_FACTOR = 1.5

    def __init__(self):
        self.received = 0
        self.missed_est = 0
        self.reordered = 0
        self.last_stamp = None
        self._deltas = deque(maxlen=self.PERIOD_WINDOW)
        self._jitter = 0.0
        self._last_transit = None

    @property
    def period_s(self):
        """Running median scan period, or None while warming up.

        Median rather than mean: one 400 ms stall would drag a mean far enough
        to hide every subsequent single-scan gap behind it.
        """
        if len(self._deltas) < self.PERIOD_MIN_SAMPLES:
            return None
        return statistics.median(self._deltas)

    def observe(self, stamp, recv_mono):
        """Account for one arrival. Returns True if it should be processed."""
        self.received += 1

        if self.last_stamp is None:
            self.last_stamp = stamp
            return True

        delta = stamp - self.last_stamp
        if delta <= 0.0:
            self.reordered += 1
            return False

        period = self.period_s
        if period and period > 0.0 and delta > self.MISS_FACTOR * period:
            self.missed_est += max(0, int(round(delta / period)) - 1)
        self._deltas.append(delta)
        self.last_stamp = stamp

        transit = recv_mono - stamp
        if self._last_transit is not None:
            d = abs(transit - self._last_transit)
            self._jitter += (d - self._jitter) * self.JITTER_GAIN
        self._last_transit = transit
        return True

    @property
    def jitter_ms(self):
        return self._jitter * 1e3

    @property
    def miss_ratio(self):
        expected = self.received + self.missed_est
        return (self.missed_est / expected) if expected else 0.0


class WifiServerNode(Node):

    tracker_name = "KF Tracker"

    def __init__(self, node_name="inf_server_wifi_node"):
        super().__init__(node_name)

        # ── Detector ────────────────────────────────────────────────────────
        self.declare_parameter("weight_file", "")
        self.declare_parameter("detector_model", "DR-SPAAM")
        self.declare_parameter("conf_thresh", 0.8)
        self.declare_parameter("stride", 1)
        self.declare_parameter("panoramic_scan", True)
        self.declare_parameter("gpu", True)

        # ── DDS link ────────────────────────────────────────────────────────
        self.declare_parameter("in_scan_topic", "/wifi/scan")
        self.declare_parameter("in_odom_topic", "/wifi/odom")
        self.declare_parameter("input_reliability", "best_effort")
        self.declare_parameter("input_depth", 1)
        self.declare_parameter("link_timeout_s", 5.0)
        self.declare_parameter("status_log_period_s", 5.0)
        self.declare_parameter("odom_history", 32)
        self.declare_parameter("odom_match_tolerance_s", 0.15)

        # ── Kalman tracker ──────────────────────────────────────────────────
        self.declare_parameter("association_threshold", 1.0)
        self.declare_parameter("max_lost_frames", 10)
        self.declare_parameter("std_a_x", 1.5)
        self.declare_parameter("std_a_y", 1.5)
        self.declare_parameter("r_laser", 0.2)
        self.declare_parameter("static_speed_threshold", 0.15)
        self.declare_parameter("static_frames_required", 15)

        # ── ROS republish ───────────────────────────────────────────────────
        self.declare_parameter("track_poses_topic", "~/track_poses")
        self.declare_parameter("markers_topic", "~/markers")
        self.declare_parameter("detections_marker_topic", "~/detection_markers")
        self.declare_parameter("scan_topic", "~/scan")
        self.declare_parameter("laser_frame_id", "laser")
        self.declare_parameter("republish_scan", True)

        gp = self.get_parameter
        self.weight_file = gp("weight_file").get_parameter_value().string_value
        self.detector_model = gp("detector_model").get_parameter_value().string_value
        self.conf_thresh = gp("conf_thresh").get_parameter_value().double_value
        self.stride = gp("stride").get_parameter_value().integer_value
        self.panoramic_scan = gp("panoramic_scan").get_parameter_value().bool_value
        self.use_gpu = gp("gpu").get_parameter_value().bool_value

        self.in_scan_topic = gp("in_scan_topic").get_parameter_value().string_value
        self.in_odom_topic = gp("in_odom_topic").get_parameter_value().string_value
        self.input_reliability = gp("input_reliability").get_parameter_value().string_value
        self.input_depth = gp("input_depth").get_parameter_value().integer_value
        self.link_timeout_s = gp("link_timeout_s").get_parameter_value().double_value
        self.status_log_period_s = gp("status_log_period_s").get_parameter_value().double_value
        self.odom_history = gp("odom_history").get_parameter_value().integer_value
        self.odom_tolerance_s = gp("odom_match_tolerance_s").get_parameter_value().double_value

        self.tracker_kwargs = dict(
            association_threshold=gp("association_threshold").get_parameter_value().double_value,
            max_lost_frames=gp("max_lost_frames").get_parameter_value().integer_value,
            std_a_x=gp("std_a_x").get_parameter_value().double_value,
            std_a_y=gp("std_a_y").get_parameter_value().double_value,
            r_laser=gp("r_laser").get_parameter_value().double_value,
            static_speed_threshold=gp("static_speed_threshold").get_parameter_value().double_value,
            static_frames_required=gp("static_frames_required").get_parameter_value().integer_value,
        )

        self.track_poses_topic = gp("track_poses_topic").get_parameter_value().string_value
        self.markers_topic = gp("markers_topic").get_parameter_value().string_value
        self.detections_marker_topic = gp("detections_marker_topic").get_parameter_value().string_value
        self.scan_topic = gp("scan_topic").get_parameter_value().string_value
        self.laser_frame_id = gp("laser_frame_id").get_parameter_value().string_value
        self.do_republish_scan = gp("republish_scan").get_parameter_value().bool_value

        if not self.weight_file:
            self.get_logger().error(
                "Parameter 'weight_file' is empty! Provide a path to a valid checkpoint.")
            raise ValueError("weight_file parameter is empty.")

        self.get_logger().info(
            f"Loading detector '{self.detector_model}' from: {self.weight_file}")
        self._detector = Detector(
            self.weight_file, model=self.detector_model, gpu=self.use_gpu,
            stride=self.stride, panoramic_scan=self.panoramic_scan,
        )
        self._process_lock = threading.Lock()

        # ── link state (one stream; see the module docstring) ───────────────
        self.tracker = self.make_tracker()
        self.arrivals = ArrivalStats()
        self._odom = deque(maxlen=self.odom_history)
        self._odom_stamps = set()
        self._odom_received = 0
        self.scans_dropped_busy = 0
        self.scans_processed = 0
        self.last_scan_time = None
        self._pending = None              # one slot: newest scan wins
        self._first_seen = None
        self._last_seen = None
        self._link_up = False
        self._scan_bins = 0

        self._scan_count = 0
        self._scans_since_log = 0
        self._inference_time_ema_s = None

        # ── publishers / subscriptions ──────────────────────────────────────
        self._track_poses_pub = self.create_publisher(PoseArray, self.track_poses_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self._detections_marker_pub = self.create_publisher(Marker, self.detections_marker_topic, 10)
        self._scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        in_qos = build_qos(self.input_reliability, self.input_depth)
        self._scan_sub = self.create_subscription(
            LaserScan, self.in_scan_topic, self._scan_callback, in_qos)
        self._odom_sub = self.create_subscription(
            Odometry, self.in_odom_topic, self._odom_callback, in_qos)

        self._work_lock = threading.Lock()
        self._work_cv = threading.Condition(self._work_lock)
        self._stop = threading.Event()
        self._workers = []
        self._start_workers()
        for w in self._workers:
            w.start()
        if self.status_log_period_s > 0.0:
            self.create_timer(self.status_log_period_s, self._log_status)
        self.create_timer(1.0, self._check_link)

        self._log_banner()

    def make_tracker(self):
        """Hook so a sibling package can swap the tracker without touching the
        transport. Mirrors the benchmark nodes, where Norfair is dropped in
        here."""
        return MultiObjectTracker(**self.tracker_kwargs)

    # ── receive path ────────────────────────────────────────────────────────

    def _scan_callback(self, msg: LaserScan):
        recv_mono = time.monotonic()
        stamp = stamp_to_sec(msg.header.stamp)
        now = time.time()

        if self._first_seen is None:
            self._first_seen = now
            self._link_up = True
            self.get_logger().info(
                f"link up: first scan on {self.in_scan_topic}, "
                f"{len(msg.ranges)} bins, frame '{msg.header.frame_id or '(unset)'}'")
        elif not self._link_up:
            self._link_up = True
            self.get_logger().info(f"link restored on {self.in_scan_topic}")
        self._last_seen = now

        if self._scan_bins and len(msg.ranges) != self._scan_bins:
            self.get_logger().warn(
                f"scan geometry changed: {self._scan_bins} -> {len(msg.ranges)} bins",
                throttle_duration_sec=10.0)
        self._scan_bins = len(msg.ranges)

        
        if stamp <= 0.0:
            self.get_logger().warn(
                "scan header stamp is zero -- loss estimate and jitter need a "
                "clock on the robot", throttle_duration_sec=30.0)

        if not self.arrivals.observe(stamp, recv_mono):
            return                                  # older than one already done

        with self._work_cv:
            if self._pending is not None:
                self.scans_dropped_busy += 1
            self._pending = (msg, stamp, recv_mono)
            self._work_cv.notify()

    def _odom_callback(self, msg: Odometry):
        self._odom_received += 1
        self._last_seen = time.time()
        stamp = stamp_to_sec(msg.header.stamp)
        if stamp in self._odom_stamps:
            return
        p = msg.pose.pose
        sample = (stamp, p.position.x, p.position.y,
                  p.orientation.z, p.orientation.w)
        self._odom_stamps.add(stamp)
        if len(self._odom) == self._odom.maxlen and self._odom:
            self._odom_stamps.discard(self._odom[0][0])
        self._odom.append(sample)

        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        self.broadcast_odom_tf(p.position.x, p.position.y, yaw, stamp)

    def odom_for(self, stamp, tolerance_s):
        best, best_dt = None, tolerance_s
        for s in list(self._odom):
            dt = abs(s[0] - stamp)
            if dt <= best_dt:
                best, best_dt = s, dt
        if best is None:
            return None
        _, x, y, qz, qw = best
        return (x, y, 2.0 * math.atan2(qz, qw))

    # ── work loop ───────────────────────────────────────────────────────────

    def _start_workers(self):
        """Build the processing threads. Called before the first scan can be
        claimed, so the pipelined variant can declare its own parameters here
        and swap in a two-stage arrangement without racing the callbacks."""
        self._workers.append(threading.Thread(target=self._worker_loop, daemon=True))

    def _worker_loop(self):
        """Detection + tracking, off the executor thread.

        Sequential variant: detect and track run inline, one after the other,
        for each scan -- the arrangement grpc_server_node uses.
        """
        while not self._stop.is_set():
            item = self._await_pending()
            if item is None:
                return
            try:
                self.process_scan(*item)
            except Exception as e:
                self.get_logger().error(f"scan processing failed: {e}",
                                        throttle_duration_sec=5.0)

    def _await_pending(self):
        """Block until a scan is waiting. None means shutdown."""
        with self._work_cv:
            while self._pending is None:
                if self._stop.is_set():
                    return None
                self._work_cv.wait(0.2)
            item, self._pending = self._pending, None
            return item

    def scan_dt(self, stamp):
        """Elapsed time since the previous processed scan, for the filter.

        A gap here is real elapsed time -- missed scans included -- so it is
        passed through rather than clamped to one scan period. Guarded against
        a clock jump on the robot, which would otherwise hand the filter a
        multi-second dt and blow every track's covariance open.
        """
        dt = 0.1
        if self.last_scan_time is not None:
            candidate = stamp - self.last_scan_time
            if 0.0 < candidate <= 2.0:
                dt = candidate
        self.last_scan_time = stamp
        return dt

    def process_scan(self, msg, stamp, recv_mono):
        dt = self.scan_dt(stamp)
        odom = self.odom_for(stamp, self.odom_tolerance_s)
        self.republish_scan(msg)

        with self._process_lock:
            dets_xy = self._detect(msg)
            dets_xy, frame_id = self._to_tracking_frame(dets_xy, odom)
            active_tracks = self.tracker.step(dt, dets_xy)

        self._publish_detections_marker(dets_xy, frame_id)
        self._publish_ros(frame_id, active_tracks)

        self.scans_processed += 1
        self._scan_count += 1
        self._scans_since_log += 1

    # ── liveness / reporting ────────────────────────────────────────────────

    def _check_link(self):
        if self._first_seen is None:
            pubs = self._scan_sub.get_publisher_count()
            if pubs == 0:
                self.get_logger().warn(
                    f"waiting on {self.in_scan_topic}: no publisher discovered. Check "
                    f"ROS_LOCALHOST_ONLY is unset here AND on the robot, "
                    f"ROS_STATIC_PEERS points the other way, ROS_DOMAIN_ID matches, "
                    f"and UDP 7400-7500 is open",
                    throttle_duration_sec=10.0)
            else:
                self.get_logger().warn(
                    f"waiting on {self.in_scan_topic}: {pubs} publisher(s) discovered "
                    f"but no sample received -- QoS mismatch? this node subscribes "
                    f"{self.input_reliability.upper()}",
                    throttle_duration_sec=10.0)
            return

        if self._link_up and time.time() - self._last_seen > self.link_timeout_s:
            self._link_up = False
            self.get_logger().warn(
                f"no sample on {self.in_scan_topic} for {self.link_timeout_s:g}s -- "
                f"robot parked, or the link is down")

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        a = self.arrivals
        period = a.period_s

        if period:
            link = (f"    rx {a.received} missed~ {a.missed_est} "
                    f"({a.miss_ratio * 100:.1f}%) reord {a.reordered} "
                    f"busy-drop {self.scans_dropped_busy} | "
                    f"jitter {a.jitter_ms:.1f} ms | period {period * 1e3:.1f} ms")
        else:
            link = (f"    rx {a.received} missed~ (warming up) reord {a.reordered} "
                    f"busy-drop {self.scans_dropped_busy} | "
                    f"jitter {a.jitter_ms:.1f} ms")

        self.get_logger().info(
            f"[inf_server_wifi] {self._scan_sub.get_publisher_count()} publisher(s) | "
            f"{rate:.1f} scans/s | {self.inference_fps:.1f} FPS ({self.detector_model}) | "
            f"{self._scan_count} total\n"
            f"{link}\n"
            f"    odom {self._odom_received} rx, {len(self._odom)} buffered | "
            f"{self._scan_bins} bins")

    def _log_banner(self):
        env = "\n".join(
            f"    {k:<32}{os.environ.get(k, '<unset>')}" for k in DISCOVERY_ENV)
        self.get_logger().info(
            f"\n"
            f"  ╔═══════════════════════════════════════════════╗\n"
            f"  ║  inf_server_wifi: DR-SPAAM + {self.tracker_name:<16} ║\n"
            f"  ╚═══════════════════════════════════════════════╝\n"
            f"  Subscribing  : {self.in_scan_topic}, {self.in_odom_topic}\n"
            f"  QoS          : {self.input_reliability}, keep-last "
            f"{max(1, self.input_depth)}, volatile\n"
            f"  Odom pairing : nearest sample within "
            f"{self.odom_tolerance_s * 1e3:.0f} ms of the scan\n"
            f"  Backpressure : none. A scan arriving while the detector is busy\n"
            f"  replaces the one waiting -- the newest scan is the only useful one.\n"
            f"  No transport of our own: DDS carries real LaserScan/Odometry, so\n"
            f"  loss is estimated from stamp gaps, not counted, and nothing is\n"
            f"  mailed back to the robot -- BEST_EFFORT has no delivery report.\n"
            f"  Discovery (process environment, not set by this node):\n"
            f"{env}"
        )

    # ── ROS output (shared with inf_server / inf_server_udp) ────────────────

    def broadcast_odom_tf(self, x, y, yaw, timestamp):
        t = TransformStamped()
        sec = int(timestamp)
        nanosec = int(round((timestamp - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        t.header.stamp.sec = sec
        t.header.stamp.nanosec = nanosec
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        half = yaw / 2.0
        t.transform.rotation.z = math.sin(half)
        t.transform.rotation.w = math.cos(half)
        self._tf_broadcaster.sendTransform(t)

    def republish_scan(self, msg):
        """Re-stamp the incoming scan into our own frame.

        The robot's frame_id names a TF only the robot's own tree has. Emitting
        it under laser_frame_id, which the launch file also anchors to
        base_link, is what makes one RViz config work for all three servers.
        """
        if not self.do_republish_scan:
            return
        out = LaserScan()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.laser_frame_id
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = msg.ranges
        out.intensities = msg.intensities
        self._scan_pub.publish(out)

    def _detect(self, msg):
        if not self._detector.is_ready():
            fov_rad = msg.angle_increment * len(msg.ranges)
            self._detector.set_laser_fov(np.rad2deg(fov_rad))
            self.get_logger().info(
                f"Dynamic LiDAR FOV configured to: {np.rad2deg(fov_rad):.2f} degrees")

        scan = np.array(msg.ranges, dtype=np.float32)
        scan[scan < msg.range_min] = 29.99
        scan[scan > msg.range_max] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        scan_phi = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment

        t0 = time.perf_counter()
        dets_xy, dets_cls, _ = self._detector(scan, scan_phi=scan_phi)
        self._record_inference_time(time.perf_counter() - t0)

        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        return dets_xy[conf_mask]

    def _record_inference_time(self, elapsed_s):
        alpha = 0.1
        if self._inference_time_ema_s is None:
            self._inference_time_ema_s = elapsed_s
        else:
            self._inference_time_ema_s = (
                alpha * elapsed_s + (1.0 - alpha) * self._inference_time_ema_s)

    @property
    def inference_fps(self):
        if not self._inference_time_ema_s:
            return 0.0
        return 1.0 / self._inference_time_ema_s

    def _to_tracking_frame(self, dets_xy, odom):
        if odom is None:
            return [(float(x), float(y)) for x, y in dets_xy], "base_link"
        ox, oy, oyaw = odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        return [(ox + c * float(x) - s * float(y), oy + s * float(x) + c * float(y))
                for x, y in dets_xy], "odom"

    def _publish_detections_marker(self, dets_xy, frame_id):
        msg = Marker()
        msg.header.frame_id = frame_id
        msg.action = Marker.ADD
        msg.ns = "dr_spaam_detections"
        msg.id = 0
        msg.type = Marker.LINE_LIST
        msg.pose.orientation.w = 1.0
        msg.scale.x = 0.03
        msg.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        msg.lifetime = Duration(seconds=0.5).to_msg()

        radius = 0.4
        angles = np.linspace(0, 2 * np.pi, 20)
        offsets = radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
        for x, y in dets_xy:
            for i in range(len(offsets) - 1):
                msg.points.append(Point(x=x + offsets[i, 0], y=y + offsets[i, 1], z=0.05))
                msg.points.append(Point(x=x + offsets[i + 1, 0], y=y + offsets[i + 1, 1], z=0.05))
        self._detections_marker_pub.publish(msg)

    def _publish_ros(self, frame_id, active_tracks):
        poses = PoseArray()
        poses.header.frame_id = frame_id
        markers = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()

        for t in active_tracks:
            vx, vy = t.velocity
            heading = math.atan2(vy, vx)

            p = Pose()
            p.position.x, p.position.y = t.position
            half = heading / 2.0
            p.orientation.z = math.sin(half)
            p.orientation.w = math.cos(half)
            poses.poses.append(p)

            cylinder = Marker()
            cylinder.header.frame_id = frame_id
            cylinder.ns = "inf_server_people"
            cylinder.id = t.id
            cylinder.type = Marker.CYLINDER
            cylinder.action = Marker.ADD
            cylinder.pose.position.x, cylinder.pose.position.y = t.position
            cylinder.pose.position.z = 0.9
            cylinder.scale = Vector3(x=0.4, y=0.4, z=1.8)
            cylinder.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.7)
            cylinder.lifetime = lifetime
            markers.markers.append(cylinder)

            if t.speed > 0.1:
                arrow = Marker()
                arrow.header.frame_id = frame_id
                arrow.ns = "inf_server_velocity"
                arrow.id = t.id
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [
                    Point(x=t.position[0], y=t.position[1], z=0.1),
                    Point(x=t.position[0] + vx, y=t.position[1] + vy, z=0.1),
                ]
                arrow.scale = Vector3(x=0.05, y=0.1, z=0.1)
                arrow.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
                arrow.lifetime = lifetime
                markers.markers.append(arrow)

        self._track_poses_pub.publish(poses)
        if active_tracks:
            self._markers_pub.publish(markers)

    def shutdown(self):
        a = self.arrivals
        self.get_logger().info(
            f"Stopping inf_server_wifi... {self.scans_processed} scans processed, "
            f"{a.received} received, ~{a.missed_est} missed, "
            f"{self.scans_dropped_busy} dropped busy")
        self._stop.set()
        with self._work_cv:
            self._work_cv.notify_all()
        for w in self._workers:
            w.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WifiServerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass          # Ctrl-C / SIGTERM are normal shutdowns, not errors
    except Exception as e:
        print(f"Error starting inf_server_wifi node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
