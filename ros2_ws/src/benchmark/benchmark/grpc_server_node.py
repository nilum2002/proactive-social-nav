import math
import os
import sys
import threading
import time
from concurrent import futures
from datetime import datetime

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped, Vector3
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster

import grpc
import psutil
import csv
import pandas as pd
from dr_spaam.detector import Detector
from benchmark.kalman_tracker import MultiObjectTracker

try:
    import pynvml
except ImportError:
    pynvml = None

# protoc emits absolute imports (`import perception_stream_pb2`)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception_stream_pb2          # noqa: E402
import perception_stream_pb2_grpc     # noqa: E402


def _stat_ms(samples, fn):
    """Reduce a list of second-valued samples to a rounded millisecond stat."""
    if not samples:
        return ""
    return round(float(fn(samples)) * 1e3, 3)


class GpuSampler:
    """NVML view of the GPU DR-SPAAM is inferencing on.

    Two different things are reported and they must not be confused when
    reading the CSV: `gpu_util_percent` is device-wide -- the fraction of the
    last NVML sample window in which *any* kernel was resident, so the desktop
    compositor counts too -- while `gpu_proc_mem_mb` / `torch_*_mb` and the
    duty cycle the node derives from inference time are attributable to this
    process alone.
    """

    def __init__(self, logger, enabled=True):
        self._logger = logger
        self._handle = None
        self._torch = None
        self._pid = os.getpid()
        self.name = "n/a"
        self.index = -1

        if not enabled:
            return
        if pynvml is None:
            logger.warn("nvidia-ml-py (pynvml) not installed -- GPU columns will be blank")
            return

        try:
            pynvml.nvmlInit()
            self.index = self._detector_device_index()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.name = name.decode() if isinstance(name, bytes) else name
        except Exception as e:
            self._handle = None
            logger.warn(f"NVML unavailable ({e}) -- GPU columns will be blank")

    def _detector_device_index(self):
        """Sample the device torch actually put the model on, not blindly GPU 0."""
        try:
            import torch
            self._torch = torch
            if torch.cuda.is_available():
                return torch.cuda.current_device()
        except Exception:
            pass
        return 0

    @property
    def available(self):
        return self._handle is not None

    @staticmethod
    def _optional(fn, scale=1.0):
        """Power/temperature/clocks are unsupported on some (especially laptop)
        GPUs; one unsupported metric must not blank out the whole row."""
        try:
            return round(fn() * scale, 2)
        except Exception:
            return ""

    def _process_gpu_mem_mb(self):
        """GPU memory NVML attributes to this PID. Reported separately from the
        device total because other processes share the card."""
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if p.pid == self._pid and p.usedGpuMemory is not None:
                    return round(p.usedGpuMemory / 1024**2, 1)
        except Exception:
            return ""
        return 0.0

    def sample(self):
        row = {
            "gpu_util_percent": "",
            "gpu_mem_util_percent": "",
            "gpu_mem_used_mb": "",
            "gpu_mem_total_mb": "",
            "gpu_proc_mem_mb": "",
            "gpu_power_w": "",
            "gpu_temp_c": "",
            "gpu_sm_clock_mhz": "",
            "torch_alloc_mb": "",
            "torch_reserved_mb": "",
            "torch_peak_mb": "",
        }

        if self.available:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                row["gpu_util_percent"] = float(util.gpu)
                row["gpu_mem_util_percent"] = float(util.memory)
                row["gpu_mem_used_mb"] = round(mem.used / 1024**2, 1)
                row["gpu_mem_total_mb"] = round(mem.total / 1024**2, 1)
                row["gpu_proc_mem_mb"] = self._process_gpu_mem_mb()
                row["gpu_power_w"] = self._optional(
                    lambda: pynvml.nvmlDeviceGetPowerUsage(self._handle), 1e-3)
                row["gpu_temp_c"] = self._optional(
                    lambda: pynvml.nvmlDeviceGetTemperature(
                        self._handle, pynvml.NVML_TEMPERATURE_GPU))
                row["gpu_sm_clock_mhz"] = self._optional(
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        self._handle, pynvml.NVML_CLOCK_SM))
            except Exception as e:
                self._logger.warn(f"NVML sample failed: {e}", throttle_duration_sec=30.0)

        # NVML sees the whole reserved pool, `memory_allocated` sees live tensors.
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                row["torch_alloc_mb"] = round(self._torch.cuda.memory_allocated() / 1024**2, 1)
                row["torch_reserved_mb"] = round(self._torch.cuda.memory_reserved() / 1024**2, 1)
                row["torch_peak_mb"] = round(self._torch.cuda.max_memory_allocated() / 1024**2, 1)
            except Exception:
                pass

        return row

    def shutdown(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None


class PerfStats:
    """Per-scan timings recorded on gRPC pool threads, drained once per report
    row. Raw samples are kept rather than an EMA so each row can carry an
    honest mean/p95/max for the interval it covers."""


    MAX_SAMPLES = 20000

    def __init__(self):
        self._lock = threading.Lock()
        self._inference_s = []
        self._predict_s = []
        self._service_s = []
        self._wire_s = []

    def record_inference(self, inference_s, predict_s=None):
        with self._lock:
            if len(self._inference_s) < self.MAX_SAMPLES:
                self._inference_s.append(inference_s)
                # Kept in its own list, not paired with the one above: a detector
                # that has not been instrumented reports nothing, and the forward
                # pass column must go blank without blanking the total as well.
                if predict_s is not None:
                    self._predict_s.append(predict_s)

    def record_service(self, service_s, wire_s):
        with self._lock:
            if len(self._service_s) < self.MAX_SAMPLES:
                self._service_s.append(service_s)
                self._wire_s.append(wire_s)

    def drain(self):
        with self._lock:
            snapshot = (self._inference_s, self._predict_s,
                        self._service_s, self._wire_s)
            self._inference_s, self._predict_s = [], []
            self._service_s, self._wire_s = [], []
        return snapshot


class PerceptionServicer(perception_stream_pb2_grpc.PerceptionServiceServicer):
    """Runs DR-SPAAM + a Kalman tracker per connected robot."""

    def __init__(self, node):
        self._node = node
        self._logger = node.get_logger()
        self._client_count = 0
        self._count_lock = threading.Lock()

    @property
    def client_count(self):
        with self._count_lock:
            return self._client_count

    def StreamSensorData(self, request_iterator, context):
        peer = context.peer()
        with self._count_lock:
            self._client_count += 1
        self._logger.info(f"robot connected: {peer} (now {self.client_count} client(s))")

        tracker = self._node.make_tracker()
        latest_odom = None   # (x, y, yaw)
        last_scan_time = None
        scans_processed = 0

        try:
            for frame in request_iterator:
                kind = frame.WhichOneof("payload")
                if kind == "odom":
                    o = frame.odom
                    yaw = 2.0 * math.atan2(o.qz, o.qw)
                    latest_odom = (o.x, o.y, yaw)
                    self._node.broadcast_odom_tf(o.x, o.y, yaw, o.timestamp)
                    continue
                if kind != "scan":
                    continue

                # Latency clocks. perf_counter drives the server-side service
                # time; the wall clock is only for the wire estimate below.
                arrival_t = time.perf_counter()
                arrival_wall = time.time()

                scan_pb = frame.scan
                dt = 0.1
                if last_scan_time is not None:
                    candidate = scan_pb.timestamp - last_scan_time
                    if 0.0 < candidate <= 2.0:
                        dt = candidate
                last_scan_time = scan_pb.timestamp

                self._node.republish_scan(scan_pb)
                self._node.process_scan(scan_pb, latest_odom, tracker, dt)

                self._node.record_service_latency(
                    service_s=time.perf_counter() - arrival_t,
                    wire_s=arrival_wall - scan_pb.timestamp,
                )
                scans_processed += 1
        finally:
            with self._count_lock:
                self._client_count -= 1
            self._logger.info(f"robot disconnected: {peer} (now {self.client_count} client(s))")

        return perception_stream_pb2.SensorAck(scans_processed=scans_processed)


class InfServerNode(Node):

    def __init__(self, node_name="inf_server_node"):
        super().__init__(node_name)

        # ── Detector parameters ─────────────────────────────────────────────
        self.declare_parameter("weight_file", "")
        self.declare_parameter("detector_model", "DR-SPAAM")
        self.declare_parameter("conf_thresh", 0.8)
        self.declare_parameter("stride", 1)
        self.declare_parameter("panoramic_scan", True)
        self.declare_parameter("gpu", True)

        # ── gRPC server parameters ──────────────────────────────────────────
        self.declare_parameter("bind_address", "0.0.0.0")
        self.declare_parameter("port", 50053)
        self.declare_parameter("max_clients", 4)
        self.declare_parameter("max_message_mb", 10)
        self.declare_parameter("status_log_period_s", 5.0)

        # ── Kalman tracker parameters ───────────────────────────────────────
        self.declare_parameter("association_threshold", 1.0)
        self.declare_parameter("max_lost_frames", 10)
        self.declare_parameter("std_a_x", 1.5)
        self.declare_parameter("std_a_y", 1.5)
        self.declare_parameter("r_laser", 0.2)
        self.declare_parameter("static_speed_threshold", 0.15)
        self.declare_parameter("static_frames_required", 15)

        # ── ROS republish parameters ────────────────────────────────────────
        self.declare_parameter("track_poses_topic", "~/track_poses")
        self.declare_parameter("markers_topic", "~/markers")
        self.declare_parameter("detections_marker_topic", "~/detection_markers")
        self.declare_parameter("scan_topic", "~/scan")
        self.declare_parameter("laser_frame_id", "laser")
        # ------------------- Service log file path (CSV) -----------------------------
        self.declare_parameter("service_log_file", "/inf_server_service_log.csv")
        self.declare_parameter("sys_report_period_s", 1.0)

        gp = self.get_parameter
        self.weight_file = gp("weight_file").get_parameter_value().string_value
        self.detector_model = gp("detector_model").get_parameter_value().string_value
        self.conf_thresh = gp("conf_thresh").get_parameter_value().double_value
        self.stride = gp("stride").get_parameter_value().integer_value
        self.panoramic_scan = gp("panoramic_scan").get_parameter_value().bool_value
        self.use_gpu = gp("gpu").get_parameter_value().bool_value

        self.bind_address = gp("bind_address").get_parameter_value().string_value
        self.port = gp("port").get_parameter_value().integer_value
        self.max_clients = gp("max_clients").get_parameter_value().integer_value
        self.max_message_mb = gp("max_message_mb").get_parameter_value().integer_value
        self.status_log_period_s = gp("status_log_period_s").get_parameter_value().double_value
        self.service_log_file = gp("service_log_file").get_parameter_value().string_value
        self.sys_report_period_s = gp("sys_report_period_s").get_parameter_value().double_value

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

        if not self.weight_file:
            self.get_logger().error("Parameter 'weight_file' is empty! Provide a path to a valid checkpoint.")
            raise ValueError("weight_file parameter is empty.")

        self.get_logger().info(f"Loading detector '{self.detector_model}' from: {self.weight_file}")
        self._detector = Detector(
            self.weight_file,
            model=self.detector_model,
            gpu=self.use_gpu,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )
        self._process_lock = threading.Lock()

        self._track_poses_pub = self.create_publisher(PoseArray, self.track_poses_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self._detections_marker_pub = self.create_publisher(Marker, self.detections_marker_topic, 10)
        self._scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._scan_count = 0
        self._scans_since_log = 0
        self._inference_time_ema_s = None

        # ── Resource / performance monitoring ───────────────────────────────
        self._perf = PerfStats()
        self._gpu = GpuSampler(self.get_logger(), enabled=self.use_gpu)
        self._proc = psutil.Process()
        self._cpu_count = psutil.cpu_count() or 1
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)

        self._report_writer = None
        self._report_file = None
        self._report_lock = threading.Lock()
        self._report_t0 = time.perf_counter()
        self._last_report_t = self._report_t0
        self._scan_count_at_report = 0
        self._open_sys_report()

        self._servicer = self._make_servicer()
        max_bytes = self.max_message_mb * 1024 * 1024
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_clients + 2),
            options=[
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ],
        )
        perception_stream_pb2_grpc.add_PerceptionServiceServicer_to_server(self._servicer, self._server)

        bind_target = f"{self.bind_address}:{self.port}"
        if self._server.add_insecure_port(bind_target) == 0:
            raise RuntimeError(f"failed to bind gRPC server to {bind_target} (port in use?)")
        self._server.start()

        if self.status_log_period_s > 0.0:
            self.create_timer(self.status_log_period_s, self._log_status)

        if self._report_writer is not None and self.sys_report_period_s > 0.0:
            self.create_timer(self.sys_report_period_s, self._write_sys_report_row)

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║{('inf_server: DR-SPAAM + ' + self.tracker_name).center(42)}║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Serving on   : {bind_target}  (PerceptionService/StreamSensorData)\n"
            f"  Max clients  : {self.max_clients}\n"
            f"  Max message  : {self.max_message_mb} MiB\n"
            f"  Expects the robot to forward its /scan and /odom topics as an\n"
            f"  interleaved SensorFrame stream (client-streaming, no reply per frame);\n"
            f"  tracks are published locally to {self.track_poses_topic} / {self.markers_topic},\n"
            f"  raw scan to {self.scan_topic}, and odom->base_link is broadcast on /tf."
        )

    # Shown in the startup banner; overridden by the Norfair variants.
    tracker_name = "KF Tracker"

    def _make_servicer(self):
        return PerceptionServicer(self)

    def make_tracker(self):
        """One tracker per connection. Overridden by the Norfair variants to
        swap the tracking backend without touching the servicers."""
        return MultiObjectTracker(**self.tracker_kwargs)

    def broadcast_odom_tf(self, x, y, yaw, timestamp):
        """Publish odom->base_link using odometry received over gRPC, so this
        server has a real TF chain of its own instead of depending on the
        robot's (unavailable, over-the-network) /tf."""
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

    def republish_scan(self, scan_pb):
        """Re-emit the scan received over gRPC as a real sensor_msgs/LaserScan,
        so it's visible in RViz2 without any dependency on the robot's own
        ROS graph being reachable."""
        msg = LaserScan()
        sec = int(scan_pb.timestamp)
        nanosec = int(round((scan_pb.timestamp - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = self.laser_frame_id
        msg.angle_min = scan_pb.angle_min
        msg.angle_max = scan_pb.angle_max
        msg.angle_increment = scan_pb.angle_increment
        msg.range_min = scan_pb.range_min
        msg.range_max = scan_pb.range_max
        msg.ranges = list(scan_pb.ranges)
        self._scan_pub.publish(msg)

    def process_scan(self, scan_pb, latest_odom, tracker, dt):
        """Run detection + tracking for one scan. Called from a gRPC pool thread."""
        with self._process_lock:
            dets_xy = self._detect(scan_pb)
            dets_xy, frame_id = self._to_tracking_frame(dets_xy, latest_odom)
            active_tracks = tracker.step(dt, dets_xy)

        self._publish_detections_marker(dets_xy, frame_id)
        self._publish_ros(frame_id, active_tracks)

        self._scan_count += 1
        self._scans_since_log += 1

    def _detect(self, scan_pb):
        if not self._detector.is_ready():
            fov_rad = scan_pb.angle_increment * len(scan_pb.ranges)
            self._detector.set_laser_fov(np.rad2deg(fov_rad))
            self.get_logger().info(f"Dynamic LiDAR FOV configured to: {np.rad2deg(fov_rad):.2f} degrees")

        scan = np.array(scan_pb.ranges, dtype=np.float32)
        scan[scan < scan_pb.range_min] = 29.99
        scan[scan > scan_pb.range_max] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        scan_phi = scan_pb.angle_min + np.arange(len(scan_pb.ranges)) * scan_pb.angle_increment

        t0 = time.perf_counter()
        dets_xy, dets_cls, _ = self._detector(scan, scan_phi=scan_phi)
        elapsed_s = time.perf_counter() - t0
        # getattr, not attribute access: an un-instrumented dr_spaam checkout is
        # still a working detector here, it just leaves the predict_ms_* columns
        # empty instead of crashing every scan.
        self._record_inference_time(
            elapsed_s, getattr(self._detector, "last_predict_s", None))

        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        return dets_xy[conf_mask]

    def _record_inference_time(self, elapsed_s, predict_s=None):
        self._perf.record_inference(elapsed_s, predict_s)
        alpha = 0.1
        if self._inference_time_ema_s is None:
            self._inference_time_ema_s = elapsed_s
        else:
            self._inference_time_ema_s = alpha * elapsed_s + (1.0 - alpha) * self._inference_time_ema_s

    @property
    def inference_fps(self):
        if not self._inference_time_ema_s:
            return 0.0
        return 1.0 / self._inference_time_ema_s

    def _to_tracking_frame(self, dets_xy, latest_odom):
        """Compensate for robot motion by projecting detections into the odom
        frame using the latest odometry; otherwise track in the raw laser frame.

        The frame depends only on whether odometry is available -- never on how
        many detections this scan happened to produce. Returning "base_link" for
        an empty scan (as this used to) mislabels the frame while tracks are
        coasting: the tracks themselves are still in odom coordinates, so RViz
        re-anchors them to base_link and they appear glued to the robot for as
        long as the detector misses. With sparse detections at a high
        conf_thresh that is most frames, and the tracks visibly drag along.
        """
        if latest_odom is None:
            return [(float(x), float(y)) for x, y in dets_xy], "base_link"

        ox, oy, oyaw = latest_odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        transformed = [
            (ox + c * float(x) - s * float(y), oy + s * float(x) + c * float(y))
            for x, y in dets_xy
        ]
        return transformed, "odom"

    def _publish_detections_marker(self, dets_xy, frame_id):
        """Red circle per raw DR-SPAAM detection, published every scan before
        Kalman confirmation -- same LINE_LIST-circle style as the original
        dr_spaam_ros2_node's rviz_marker, so a person shows up here the instant
        DR-SPAAM sees them, before the tracker has confirmed a stable track."""
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

    # ── Resource / performance report ───────────────────────────────────────

    def record_service_latency(self, service_s, wire_s):
        """Called from the gRPC pool thread once a scan is fully handled."""
        self._perf.record_service(service_s, wire_s)

    @staticmethod
    def _find_report_dir():
        """sys_reports_server lives at the repo root, but after a colcon build
        this module runs from ros2_ws/install/... -- so the root is found by
        walking up from this file rather than trusting the cwd."""
        d = os.path.dirname(os.path.abspath(__file__))
        while True:
            if os.path.isdir(os.path.join(d, "sys_reports_server")) or \
               os.path.isdir(os.path.join(d, ".git")):
                return os.path.join(d, "sys_reports_server")
            parent = os.path.dirname(d)
            if parent == d:
                return os.path.join(os.getcwd(), "sys_reports_server")
            d = parent

    def _open_sys_report(self):
        """Open the CSV named by the service_log_file parameter. The header is
        written up front so a run that is killed mid-stream still leaves a
        readable file."""
        # basename() because the parameter carries a leading '/' -- it names the
        # file inside sys_reports_server, it is not an absolute path.
        name = os.path.basename(self.service_log_file) or "inf_server_service_log.csv"
        report_dir = self._find_report_dir()
        self._report_path = os.path.join(report_dir, name)

        try:
            os.makedirs(report_dir, exist_ok=True)
            existed = os.path.exists(self._report_path)
            self._report_file = open(self._report_path, "w", newline="")
            self._report_writer = csv.DictWriter(
                self._report_file, fieldnames=self._report_fieldnames())
            self._report_writer.writeheader()
            self._report_file.flush()
        except OSError as e:
            self._report_writer = None
            self.get_logger().error(f"cannot write sys report to {self._report_path}: {e}")
            return

        if existed:
            self.get_logger().warn(
                f"overwriting previous report {self._report_path} "
                f"(change service_log_file in params.yaml to keep both runs)")

        gpu_desc = f"{self._gpu.name} (cuda:{self._gpu.index})" if self._gpu.available else "none"
        self.get_logger().info(
            f"sys report -> {self._report_path} "
            f"every {self.sys_report_period_s:g}s | monitored GPU: {gpu_desc}")

    @staticmethod
    def _report_fieldnames():
        return [
            "wall_time", "elapsed_s", "clients",
            # DR-SPAAM throughput
            "scans_total", "scans_in_window", "scan_rate_hz", "drspaam_fps",
            # inference_ms_* is the whole Detector.__call__ (cutout preprocessing
            # + forward + NMS); predict_ms_* is the network forward pass alone.
            # The difference between the two means is the CPU-side work.
            "inference_ms_mean", "inference_ms_p95", "inference_ms_max",
            "predict_ms_mean", "predict_ms_p95", "predict_ms_max",
            # gRPC latency
            "grpc_service_ms_mean", "grpc_service_ms_p95", "grpc_service_ms_max",
            "grpc_wire_ms_mean", "grpc_wire_ms_p95",
            # GPU
            "gpu_util_percent", "drspaam_gpu_duty_percent", "gpu_mem_util_percent",
            "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_proc_mem_mb",
            "torch_alloc_mb", "torch_reserved_mb", "torch_peak_mb",
            "gpu_power_w", "gpu_temp_c", "gpu_sm_clock_mhz",
            # CPU / RAM / swap
            "cpu_percent_system", "cpu_percent_process", "proc_threads",
            "ram_used_mb", "ram_total_mb", "ram_percent", "proc_rss_mb",
            "swap_used_mb", "swap_total_mb", "swap_percent",
        ]

    def _write_sys_report_row(self):
        now = time.perf_counter()
        window_s = now - self._last_report_t
        if window_s <= 0.0:
            return
        self._last_report_t = now

        inference_s, predict_s, service_s, wire_s = self._perf.drain()
        scans = self._scan_count - self._scan_count_at_report
        self._scan_count_at_report = self._scan_count

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        row = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(now - self._report_t0, 3),
            "clients": self._servicer.client_count,
            "scans_total": self._scan_count,
            "scans_in_window": scans,
            "scan_rate_hz": round(scans / window_s, 2),
            # Throughput the detector alone could sustain (1 / mean forward pass),
            # which is above scan_rate_hz whenever the robot feeds scans slower
            # than DR-SPAAM can consume them. Blank rather than 0 while no scans
            # are arriving, so idle intervals do not drag the run averages down.
            "drspaam_fps": round(len(inference_s) / sum(inference_s), 2) if inference_s else "",
            "inference_ms_mean": _stat_ms(inference_s, np.mean),
            "inference_ms_p95": _stat_ms(inference_s, lambda a: np.percentile(a, 95)),
            "inference_ms_max": _stat_ms(inference_s, np.max),
            # Network forward pass only, GPU-synchronised inside the detector.
            # Blank when running against a dr_spaam that does not expose it.
            "predict_ms_mean": _stat_ms(predict_s, np.mean),
            "predict_ms_p95": _stat_ms(predict_s, lambda a: np.percentile(a, 95)),
            "predict_ms_max": _stat_ms(predict_s, np.max),
            "grpc_service_ms_mean": _stat_ms(service_s, np.mean),
            "grpc_service_ms_p95": _stat_ms(service_s, lambda a: np.percentile(a, 95)),
            "grpc_service_ms_max": _stat_ms(service_s, np.max),
            "grpc_wire_ms_mean": _stat_ms(wire_s, np.mean),
            "grpc_wire_ms_p95": _stat_ms(wire_s, lambda a: np.percentile(a, 95)),
            # Share of wall-clock time spent inside DR-SPAAM forward passes.
            # Unlike gpu_util_percent this excludes every other GPU client on
            # the machine, so it is the detector's own load on the card -- but
            # read it as an upper bound: the timed call also covers the CPU-side
            # pre/post-processing around the kernels, so it can sit above the
            # device-wide util figure.
            "drspaam_gpu_duty_percent": round(100.0 * sum(inference_s) / window_s, 2)
                                        if inference_s else "",
            "cpu_percent_system": psutil.cpu_percent(interval=None),
            # Can exceed 100%: it is summed over cores, not normalised.
            "cpu_percent_process": round(self._proc.cpu_percent(interval=None), 1),
            "proc_threads": self._proc.num_threads(),
            "ram_used_mb": round(vm.used / 1024**2, 1),
            "ram_total_mb": round(vm.total / 1024**2, 1),
            "ram_percent": vm.percent,
            "proc_rss_mb": round(self._proc.memory_info().rss / 1024**2, 1),
            "swap_used_mb": round(sw.used / 1024**2, 1),
            "swap_total_mb": round(sw.total / 1024**2, 1),
            "swap_percent": sw.percent,
        }
        row.update(self._gpu.sample())

        with self._report_lock:
            if self._report_writer is None:
                return
            self._report_writer.writerow(row)
            # Flushed every row: these runs are ended with Ctrl-C, and buffered
            # rows would be lost exactly when the interesting part just happened.
            self._report_file.flush()

    def _close_sys_report(self):
        with self._report_lock:
            if self._report_file is None:
                return
            self._report_writer = None
            self._report_file.close()
            self._report_file = None
        self._write_sys_summary()

    def _write_sys_summary(self):
        """Condense the run into a one-file summary next to the raw CSV, so the
        sequential and pipelined runs can be compared without re-deriving the
        same aggregates by hand every time."""
        stem, ext = os.path.splitext(self._report_path)
        summary_path = f"{stem}_summary{ext}"
        try:
            df = pd.read_csv(self._report_path)
            if df.empty:
                self.get_logger().warn("sys report is empty -- no summary written")
                return
            summary = df.select_dtypes("number").describe().T
            summary.index.name = "metric"
            summary.to_csv(summary_path)
        except Exception as e:
            self.get_logger().warn(f"could not write sys summary: {e}")
            return

        def avg(col):
            return df[col].mean() if col in df and df[col].notna().any() else float("nan")

        # Derived from the mean forward pass rather than averaging the per-window
        # FPS column, so the two figures printed side by side agree (the mean of
        # 1/t is not 1/mean(t)).
        mean_inference_ms = avg("inference_ms_mean")
        fps = 1e3 / mean_inference_ms if mean_inference_ms else float("nan")

        self.get_logger().info(
            f"\n  sys report : {self._report_path}\n"
            f"  summary    : {summary_path}\n"
            f"  duration {df['elapsed_s'].iloc[-1]:.1f}s over {len(df)} samples, "
            f"{int(df['scans_total'].iloc[-1])} scans\n"
            f"  DR-SPAAM   {fps:.1f} FPS | "
            f"{mean_inference_ms:.2f} ms/scan | "
            f"GPU duty {avg('drspaam_gpu_duty_percent'):.1f}%\n"
            f"             of which forward pass {avg('predict_ms_mean'):.2f} ms, "
            f"pre/post {mean_inference_ms - avg('predict_ms_mean'):.2f} ms\n"
            f"  gRPC       service {avg('grpc_service_ms_mean'):.2f} ms | "
            f"p95 {avg('grpc_service_ms_p95'):.2f} ms\n"
            f"  GPU        util {avg('gpu_util_percent'):.1f}% | "
            f"mem {avg('gpu_mem_used_mb'):.0f} MB | {avg('gpu_power_w'):.1f} W\n"
            f"  Host       CPU {avg('cpu_percent_system'):.1f}% | "
            f"RAM {avg('ram_percent'):.1f}% | swap {avg('swap_percent'):.1f}%"
        )

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        self.get_logger().info(
            f"[inf_server] {self._servicer.client_count} client(s) | "
            f"{rate:.1f} scans/s (pipeline) | {self.inference_fps:.1f} FPS (DR-SPAAM) | "
            f"{self._scan_count} total"
        )

    def shutdown(self):
        self.get_logger().info("Stopping gRPC server...")
        self._server.stop(grace=1.0).wait()
        # After the server is down, so no pool thread can still be recording.
        self._close_sys_report()
        self._gpu.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = InfServerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting inf_server node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
