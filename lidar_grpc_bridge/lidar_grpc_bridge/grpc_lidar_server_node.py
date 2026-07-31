"""grpc_lidar_server_node.py.

Bridges ROS2 sensor_msgs/LaserScan onto the gRPC LidarService defined in
proto/lidar_stream.proto, so an off-board machine (e.g. the inference laptop at
192.168.0.196) can pull the live scan stream without needing ROS2 on the network.

Direction of travel: StreamScan is a SERVER-streaming RPC, so the machine holding
the lidar — this node — is the gRPC server, and the consumer dials in and pulls.
That is the same contract jetson_grpc_client.py already speaks; point its RPI_IP at
whichever host runs this node.

Threading: grpc.server runs its own thread pool, entirely separate from the rclpy
executor. The ROS subscription callback (executor thread) hands scans to the
streaming RPC handlers (pool threads) through one small queue per connected client.
"""
import os
import queue
import sys
import threading
from concurrent import futures

import math

import grpc
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry

from proactive_nav_msgs.msg import Track as TrackMsg, TrackArray as TrackArrayMsg

# The generated stubs use absolute imports (`import lidar_stream_pb2`) because that is
# what protoc emits, and jetson_grpc_client.py imports them the same way. Rather than
# hand-editing generated code — which would make every regeneration a conflict — put
# this package's own directory on sys.path so those absolute imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_stream_pb2          # noqa: E402
import lidar_stream_pb2_grpc     # noqa: E402
import track_stream_pb2          # noqa: E402
import track_stream_pb2_grpc     # noqa: E402
import odom_stream_pb2           # noqa: E402
import odom_stream_pb2_grpc      # noqa: E402


class LidarServicer(lidar_stream_pb2_grpc.LidarServiceServicer):
    """Fans the newest scan out to every currently connected client."""

    def __init__(self, logger, poll_timeout_s=0.5):
        self._logger = logger
        self._poll_timeout_s = poll_timeout_s
        self._client_queues = []
        self._lock = threading.Lock()

    @property
    def client_count(self):
        with self._lock:
            return len(self._client_queues)

    def broadcast(self, scan_pb):
        """Hand a scan to every client. Called from the ROS executor thread."""
        with self._lock:
            queues = list(self._client_queues)

        for q in queues:
            try:
                q.put_nowait(scan_pb)
            except queue.Full:
                # Latest-wins. A client slower than the lidar (or on a stalled TCP
                # connection) must not build an unbounded backlog: stale scans are
                # useless for tracking, and queuing them would grow memory without
                # bound and hand the consumer data that is seconds old. Drop the
                # pending scan and replace it with this one.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(scan_pb)
                except queue.Full:
                    pass

    def StreamScan(self, request, context):
        """Server-streaming RPC: yields scans until the client goes away."""
        q = queue.Queue(maxsize=1)
        with self._lock:
            self._client_queues.append(q)
        peer = context.peer()
        self._logger.info(f"gRPC client connected: {peer} (now {self.client_count} client(s))")

        try:
            while context.is_active():
                try:
                    scan_pb = q.get(timeout=self._poll_timeout_s)
                except queue.Empty:
                    # Time out rather than block forever so is_active() gets re-checked
                    # and a client that vanished without a clean half-close is reaped
                    # instead of pinning a pool thread until the next scan arrives.
                    continue
                yield scan_pb
        finally:
            with self._lock:
                if q in self._client_queues:
                    self._client_queues.remove(q)
            self._logger.info(f"gRPC client disconnected: {peer} (now {self.client_count} client(s))")


class OdometryServicer(odom_stream_pb2_grpc.OdometryServiceServicer):
    """Fans the newest odom->base_link pose out to every currently connected client.

    Same broadcast/latest-wins shape as LidarServicer: this exists so RViz/tf2 on the
    laptop get the robot's pose over the same gRPC link as the scan, instead of
    depending on native ROS2 DDS discovery reaching across the network.
    """

    def __init__(self, logger, poll_timeout_s=0.5):
        self._logger = logger
        self._poll_timeout_s = poll_timeout_s
        self._client_queues = []
        self._lock = threading.Lock()

    @property
    def client_count(self):
        with self._lock:
            return len(self._client_queues)

    def broadcast(self, odom_pb):
        with self._lock:
            queues = list(self._client_queues)

        for q in queues:
            try:
                q.put_nowait(odom_pb)
            except queue.Full:
                # Latest-wins, same reasoning as LidarServicer.broadcast: a stale pose
                # is worse than a late-but-current one for TF/RViz.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(odom_pb)
                except queue.Full:
                    pass

    def StreamOdometry(self, request, context):
        q = queue.Queue(maxsize=1)
        with self._lock:
            self._client_queues.append(q)
        peer = context.peer()
        self._logger.info(f"gRPC odom client connected: {peer} (now {self.client_count} client(s))")

        try:
            while context.is_active():
                try:
                    odom_pb = q.get(timeout=self._poll_timeout_s)
                except queue.Empty:
                    continue
                yield odom_pb
        finally:
            with self._lock:
                if q in self._client_queues:
                    self._client_queues.remove(q)
            self._logger.info(f"gRPC odom client disconnected: {peer} (now {self.client_count} client(s))")


class TrackPushServicer(track_stream_pb2_grpc.TrackServiceServicer):
    """Return path: the laptop pushes tracked-person state up this same server.

    Client-streaming rather than a second server on the laptop — the laptop is
    already dialing this process to pull scans (LidarService.StreamScan), so it opens
    a second call here instead of standing up its own gRPC server + port. Each
    incoming TrackArray is handed to on_track_array(pb) as it arrives; the caller
    (LidarGrpcServerNode) does the ROS republish so this class stays gRPC-only.
    """

    def __init__(self, logger, on_track_array):
        self._logger = logger
        self._on_track_array = on_track_array

    def PushTracks(self, request_iterator, context):
        peer = context.peer()
        self._logger.info(f"track push connected: {peer}")
        count = 0
        try:
            for track_array_pb in request_iterator:
                self._on_track_array(track_array_pb)
                count += len(track_array_pb.tracks)
            return track_stream_pb2.PushAck(tracks_received=count)
        finally:
            self._logger.info(f"track push disconnected: {peer} ({count} track(s) total)")


class LidarGrpcServerNode(Node):

    def __init__(self):
        super().__init__("lidar_grpc_server_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("bind_address", "0.0.0.0")
        self.declare_parameter("port", 50051)
        self.declare_parameter("max_clients", 4)
        self.declare_parameter("max_message_mb", 10)
        self.declare_parameter("status_log_period_s", 5.0)
        # Return path: where pushed tracks get republished locally for the planner.
        self.declare_parameter("tracks_topic", "~/tracks")
        self.declare_parameter("track_poses_topic", "~/track_poses")
        self.declare_parameter("track_frame_id", "")  # "" = keep what the pusher sent
        self.declare_parameter("use_source_track_timestamp", True)
        # Odometry: streamed to the consumer over gRPC (OdometryService/StreamOdometry)
        # instead of relying on native ROS2 DDS discovery to carry /tf across the
        # network -- same reasoning as the scan going over gRPC instead of a topic.
        self.declare_parameter("enable_odom", True)
        self.declare_parameter("odom_topic", "/odom")

        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.bind_address = self.get_parameter("bind_address").get_parameter_value().string_value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.max_clients = self.get_parameter("max_clients").get_parameter_value().integer_value
        self.max_message_mb = self.get_parameter("max_message_mb").get_parameter_value().integer_value
        self.status_log_period_s = self.get_parameter("status_log_period_s").get_parameter_value().double_value
        self.tracks_topic = self.get_parameter("tracks_topic").get_parameter_value().string_value
        self.track_poses_topic = self.get_parameter("track_poses_topic").get_parameter_value().string_value
        self.track_frame_id_override = self.get_parameter("track_frame_id").get_parameter_value().string_value
        self.use_source_track_timestamp = self.get_parameter("use_source_track_timestamp").get_parameter_value().bool_value
        self.enable_odom = self.get_parameter("enable_odom").get_parameter_value().bool_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self._scan_count = 0
        self._scans_since_log = 0
        self._track_msg_count = 0
        self._track_msgs_since_log = 0
        self._odom_count = 0
        self._odom_since_log = 0

        self._servicer = LidarServicer(self.get_logger())
        self._track_servicer = TrackPushServicer(self.get_logger(), self._on_track_array)
        self._tracks_pub = self.create_publisher(TrackArrayMsg, self.tracks_topic, 10)
        self._track_poses_pub = self.create_publisher(PoseArray, self.track_poses_topic, 10)
        if self.enable_odom:
            self._odom_servicer = OdometryServicer(self.get_logger())

        # Each in-flight server-streaming call occupies one pool thread for its entire
        # lifetime, so the pool must be larger than the number of clients or the next
        # client silently hangs waiting for a free worker instead of being served.
        max_bytes = self.max_message_mb * 1024 * 1024
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_clients + 2),
            options=[
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ],
        )
        lidar_stream_pb2_grpc.add_LidarServiceServicer_to_server(self._servicer, self._server)
        track_stream_pb2_grpc.add_TrackServiceServicer_to_server(self._track_servicer, self._server)
        if self.enable_odom:
            odom_stream_pb2_grpc.add_OdometryServiceServicer_to_server(self._odom_servicer, self._server)

        bind_target = f"{self.bind_address}:{self.port}"
        bound_port = self._server.add_insecure_port(bind_target)
        if bound_port == 0:
            raise RuntimeError(f"failed to bind gRPC server to {bind_target} (port in use?)")
        self._server.start()

        # BEST_EFFORT: matches the rplidar node's sensor-data QoS and is still compatible
        # with the RELIABLE ldlidar_stl_ros2 publisher (a RELIABLE offer satisfies a
        # BEST_EFFORT request), so this bridge works against either driver unchanged.
        self._scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data
        )
        if self.enable_odom:
            # Default (RELIABLE) QoS: matches nav_msgs/Odometry publishers such as
            # kobuki_base_node, which uses the default depth-10 RELIABLE publisher.
            self._odom_sub = self.create_subscription(
                Odometry, self.odom_topic, self._odom_callback, 10
            )

        if self.status_log_period_s > 0.0:
            self._status_timer = self.create_timer(self.status_log_period_s, self._log_status)

        services_line = "LidarService/StreamScan, TrackService/PushTracks"
        if self.enable_odom:
            services_line += ", OdometryService/StreamOdometry"
        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║        LiDAR → gRPC Stream Bridge        ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Subscribed to : {self.scan_topic}"
            + (f", {self.odom_topic}" if self.enable_odom else "") + "\n"
            f"  Serving on    : {bind_target}  ({services_line})\n"
            f"  Max clients   : {self.max_clients}\n"
            f"  Max message   : {self.max_message_mb} MiB\n"
            f"  Republishing tracks pushed by the inference host to:\n"
            f"    {self.tracks_topic}  (proactive_nav_msgs/TrackArray)\n"
            f"    {self.track_poses_topic}  (geometry_msgs/PoseArray, heading in orientation)\n"
            f"  Consumers dial in and pull; e.g. set RPI_IP in jetson_grpc_client.py\n"
            f"  to this host's address."
        )

    def _scan_callback(self, msg: LaserScan):
        stamp = msg.header.stamp
        scan_pb = lidar_stream_pb2.LaserScanData(
            timestamp=stamp.sec + stamp.nanosec * 1e-9,
            angle_min=msg.angle_min,
            angle_max=msg.angle_max,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max,
            # Forwarded verbatim, NaN/inf included — the bridge must not silently
            # reinterpret no-return beams; that is the consumer's decision.
            ranges=msg.ranges,
        )
        self._servicer.broadcast(scan_pb)
        self._scan_count += 1
        self._scans_since_log += 1

    def _odom_callback(self, msg: Odometry):
        stamp = msg.header.stamp
        odom_pb = odom_stream_pb2.OdometryData(
            timestamp=stamp.sec + stamp.nanosec * 1e-9,
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            qz=msg.pose.pose.orientation.z,
            qw=msg.pose.pose.orientation.w,
            vx=msg.twist.twist.linear.x,
            vtheta=msg.twist.twist.angular.z,
        )
        self._odom_servicer.broadcast(odom_pb)
        self._odom_count += 1
        self._odom_since_log += 1

    def _on_track_array(self, pb):
        """Called from a gRPC pool thread (inside PushTracks) for each incoming
        TrackArray. rclpy publishers are thread-safe, so this publishes directly
        rather than bouncing through a queue back to the executor thread."""
        msg = TrackArrayMsg()
        if self.use_source_track_timestamp and pb.timestamp > 0.0:
            sec = int(pb.timestamp)
            nanosec = int(round((pb.timestamp - sec) * 1e9))
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.track_frame_id_override or pb.frame_id

        poses = PoseArray()
        poses.header = msg.header

        for t in pb.tracks:
            tm = TrackMsg()
            tm.id, tm.x, tm.y = t.id, t.x, t.y
            tm.vx, tm.vy, tm.speed, tm.heading = t.vx, t.vy, t.speed, t.heading
            tm.moving = t.moving
            msg.tracks.append(tm)

            self.get_logger().info(
                f"track {t.id}: pos=({t.x:+.2f}, {t.y:+.2f}) m  "
                f"vel=({t.vx:+.2f}, {t.vy:+.2f}) m/s  "
                f"dir={math.degrees(t.heading):+.1f}°  "
                f"{'moving' if t.moving else 'stationary'}"
            )

            p = Pose()
            p.position.x, p.position.y = float(t.x), float(t.y)
            half = float(t.heading) / 2.0
            p.orientation.z = math.sin(half)
            p.orientation.w = math.cos(half)
            poses.poses.append(p)

        self._tracks_pub.publish(msg)
        self._track_poses_pub.publish(poses)
        self._track_msg_count += 1
        self._track_msgs_since_log += 1

    def _log_status(self):
        scan_rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        track_rate = self._track_msgs_since_log / self.status_log_period_s
        self._track_msgs_since_log = 0
        line = (
            f"[gRPC] {self._servicer.client_count} scan client(s) | "
            f"{scan_rate:.1f} scans/s out | {self._scan_count} total   ||   "
            f"{track_rate:.1f} track msgs/s in | {self._track_msg_count} total"
        )
        if self.enable_odom:
            odom_rate = self._odom_since_log / self.status_log_period_s
            self._odom_since_log = 0
            line += (
                f"   ||   {self._odom_servicer.client_count} odom client(s) | "
                f"{odom_rate:.1f} odom/s out | {self._odom_count} total"
            )
        self.get_logger().info(line)

    def shutdown(self):
        self.get_logger().info("Stopping gRPC server...")
        self._server.stop(grace=1.0).wait()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LidarGrpcServerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting gRPC lidar server node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
