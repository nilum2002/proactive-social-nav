"""inf_server: gRPC inference server for pedestrian detection + tracking.
"""
import math
import os
import sys
import threading
import time
from concurrent import futures

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

from dr_spaam.detector import Detector
from inf_server.kalman_tracker import MultiObjectTracker

# protoc emits absolute imports (`import perception_stream_pb2`)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception_stream_pb2          # noqa: E402
import perception_stream_pb2_grpc     # noqa: E402


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

        tracker = MultiObjectTracker(**self._node.tracker_kwargs)
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

                scan_pb = frame.scan
                dt = 0.1
                if last_scan_time is not None:
                    candidate = scan_pb.timestamp - last_scan_time
                    if 0.0 < candidate <= 2.0:
                        dt = candidate
                last_scan_time = scan_pb.timestamp

                self._node.republish_scan(scan_pb)
                self._node.process_scan(scan_pb, latest_odom, tracker, dt)
                scans_processed += 1
        finally:
            with self._count_lock:
                self._client_count -= 1
            self._logger.info(f"robot disconnected: {peer} (now {self.client_count} client(s))")

        return perception_stream_pb2.SensorAck(scans_processed=scans_processed)


class InfServerNode(Node):

    def __init__(self):
        super().__init__("inf_server_node")

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
        self.declare_parameter("scan_topic", "~/scan")
        # Must match the base_link->laser static transform published alongside
        # this node (see inf_server.launch.py) -- that's the robot's physical
        # mount offset, which this server has no other way to learn since it
        # never receives the robot's own TF, only raw sensor data over gRPC.
        self.declare_parameter("laser_frame_id", "laser")

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
        # Detector.__call__ mutates internal scan-angle state, and a shared
        # MultiObjectTracker per call would race the same way, so scan processing
        # for one connection at a time is serialized here.
        self._process_lock = threading.Lock()

        self._track_poses_pub = self.create_publisher(PoseArray, self.track_poses_topic, 10)
        self._markers_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self._scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._scan_count = 0
        self._scans_since_log = 0
        # EMA of the DR-SPAAM forward-pass wall time itself (not the whole
        # pipeline) -- this is what "FPS" means for the model specifically,
        # decoupled from any gRPC/queueing time around it.
        self._inference_time_ema_s = None

        self._servicer = PerceptionServicer(self)
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

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║     inf_server: DR-SPAAM + KF Tracker    ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Serving on   : {bind_target}  (PerceptionService/StreamSensorData)\n"
            f"  Max clients  : {self.max_clients}\n"
            f"  Max message  : {self.max_message_mb} MiB\n"
            f"  Expects the robot to forward its /scan and /odom topics as an\n"
            f"  interleaved SensorFrame stream (client-streaming, no reply per frame);\n"
            f"  tracks are published locally to {self.track_poses_topic} / {self.markers_topic},\n"
            f"  raw scan to {self.scan_topic}, and odom->base_link is broadcast on /tf."
        )

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
        self._record_inference_time(time.perf_counter() - t0)

        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        return dets_xy[conf_mask]

    def _record_inference_time(self, elapsed_s):
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
        frame using the latest odometry; otherwise track in the raw laser frame."""
        if latest_odom is None or len(dets_xy) == 0:
            return [(float(x), float(y)) for x, y in dets_xy], "base_link"

        ox, oy, oyaw = latest_odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        transformed = [
            (ox + c * float(x) - s * float(y), oy + s * float(x) + c * float(y))
            for x, y in dets_xy
        ]
        return transformed, "odom"

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
