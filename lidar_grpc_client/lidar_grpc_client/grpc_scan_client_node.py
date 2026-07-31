"""grpc_scan_client_node.py.

The receiving half of the gRPC lidar link. Runs on the inference host, dials the
bridge running on the robot, pulls the LaserScanData stream, and republishes it as a
normal ROS2 sensor_msgs/LaserScan on this machine.

Why republish instead of running DR-SPAAM inline (as jetson_grpc_client.py does):
dr_spaam_ros2_node already implements FOV cropping, NaN/inf handling, the confidence
filter and the detector lifecycle, and dr_spaam_tracker_node implements the KF. Doing
inference here would mean maintaining a second copy of that preprocessing -- which is
exactly how jetson_grpc_client.py ended up missing the NaN guard that the ROS node
has. With the scan back on a topic, the whole existing pipeline (detector, tracker,
RViz markers) runs unmodified.

Return path: this node also PUSHES tracked-person state (position, velocity, heading)
back up to the same robot it pulls scans from, over TrackService.PushTracks — a second
call to the same server/port rather than a separate gRPC server+client pair. It
subscribes to dr_spaam_tracker_node's TrackArray output (which runs on this same
machine) and streams it up. The two directions run as independent threads with
independent reconnect loops (two TCP connections to one target) rather than sharing
one channel object, so a stall pulling scans cannot block pushing tracks or vice versa,
and neither reconnect loop needs to coordinate with the other.

Odometry: this node also PULLS the robot's odom->base_link pose over
OdometryService.StreamOdometry — a third call to the same server/port — and
republishes it as nav_msgs/Odometry plus a tf broadcast, exactly matching what
kobuki_base_node publishes locally on the robot. This exists so the pose reaches
RViz/tf2 on this machine over the same gRPC link as the scan, instead of depending on
native ROS2 DDS discovery to carry /tf across the network — the whole reason the scan
itself goes over gRPC rather than a plain ROS2 topic.

Threading: the blocking gRPC receive loop owns a daemon thread; rclpy.spin() keeps the
main thread. rclpy publishers are thread-safe, so the receive thread publishes directly.
"""
import os
import queue
import sys
import threading
import time

import grpc
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from proactive_nav_msgs.msg import TrackArray as TrackArrayMsg

# Reuse the generated stubs from lidar_grpc_bridge rather than keeping a second copy
# that can drift from the .proto. They use absolute imports (`import lidar_stream_pb2`)
# because that is what protoc emits, so put their directory on sys.path to resolve it.
import lidar_grpc_bridge  # noqa: F401  (imported for its __file__ location)

sys.path.insert(0, os.path.dirname(os.path.abspath(lidar_grpc_bridge.__file__)))
import lidar_stream_pb2          # noqa: E402
import lidar_stream_pb2_grpc     # noqa: E402
import track_stream_pb2          # noqa: E402
import track_stream_pb2_grpc     # noqa: E402
import odom_stream_pb2           # noqa: E402
import odom_stream_pb2_grpc      # noqa: E402


class GrpcScanClientNode(Node):

    def __init__(self):
        super().__init__("grpc_scan_client_node")

        # Address of the machine running lidar_grpc_bridge (the one holding the lidar).
        self.declare_parameter("server_address", "192.168.0.200")
        self.declare_parameter("server_port", 50051)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("frame_id", "base_laser")
        self.declare_parameter("reconnect_delay_s", 2.0)
        self.declare_parameter("max_message_mb", 10)
        self.declare_parameter("status_log_period_s", 5.0)
        # Keep the robot's original scan timestamp instead of stamping on arrival.
        # dr_spaam_tracker_node derives dt from header.stamp, and velocity is
        # (position delta / dt), so stamping locally would fold network jitter straight
        # into the velocity estimate -- the exact error the tracker's sensor-stamp dt
        # was introduced to avoid. Set false only if the two machines' clocks are far
        # enough apart that TF/RViz reject the scans as too old.
        self.declare_parameter("use_source_timestamp", True)

        # Return path (see module docstring): push tracks up on a second call to the
        # same server/port instead of running a separate gRPC server on this machine.
        self.declare_parameter("enable_track_push", True)
        self.declare_parameter("track_array_topic", "/dr_spaam_tracker_node/track_array")

        # Odometry: pulled over gRPC on a third call to the same server/port, instead
        # of relying on native ROS2 DDS discovery to carry kobuki_base_node's /tf
        # across the network. Republished here exactly as kobuki_base_node publishes
        # it locally, so dr_spaam_tracker_node's odom-frame lookup_transform() and
        # RViz see the same odom->base_link transform either way.
        self.declare_parameter("enable_odom", True)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("publish_odom_tf", True)

        self.server_address = self.get_parameter("server_address").get_parameter_value().string_value
        self.server_port = self.get_parameter("server_port").get_parameter_value().integer_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.reconnect_delay_s = self.get_parameter("reconnect_delay_s").get_parameter_value().double_value
        self.max_message_mb = self.get_parameter("max_message_mb").get_parameter_value().integer_value
        self.status_log_period_s = self.get_parameter("status_log_period_s").get_parameter_value().double_value
        self.use_source_timestamp = self.get_parameter("use_source_timestamp").get_parameter_value().bool_value
        self.enable_track_push = self.get_parameter("enable_track_push").get_parameter_value().bool_value
        self.track_array_topic = self.get_parameter("track_array_topic").get_parameter_value().string_value
        self.enable_odom = self.get_parameter("enable_odom").get_parameter_value().bool_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.odom_frame_id = self.get_parameter("odom_frame_id").get_parameter_value().string_value
        self.base_frame_id = self.get_parameter("base_frame_id").get_parameter_value().string_value
        self.publish_odom_tf = self.get_parameter("publish_odom_tf").get_parameter_value().bool_value

        # Default (RELIABLE) QoS, deliberately NOT qos_profile_sensor_data:
        # dr_spaam_ros2_node subscribes with plain depth-10 default QoS, i.e. RELIABLE.
        # A BEST_EFFORT publisher cannot satisfy a RELIABLE subscriber, so the two would
        # silently never connect. This hop is loopback on one machine, so reliability
        # costs nothing here -- the lossy link is upstream, on the gRPC side.
        self._scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)

        self._target = f"{self.server_address}:{self.server_port}"
        self._scan_count = 0
        self._scans_since_log = 0
        self._connected = False
        self._stop = threading.Event()

        if self.status_log_period_s > 0.0:
            self._status_timer = self.create_timer(self.status_log_period_s, self._log_status)

        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._rx_thread.start()

        odom_status_line = "disabled (enable_odom:=false)"
        if self.enable_odom:
            self._odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
            if self.publish_odom_tf:
                self._tf_broadcaster = TransformBroadcaster(self)
            self._odom_count = 0
            self._odom_since_log = 0
            self._odom_connected = False
            self._odom_stop = threading.Event()
            self._odom_rx_thread = threading.Thread(target=self._odom_receive_loop, daemon=True)
            self._odom_rx_thread.start()
            odom_status_line = (
                f"OdometryService/StreamOdometry -> {self.odom_topic}"
                + (f" + tf({self.odom_frame_id}->{self.base_frame_id})" if self.publish_odom_tf else "")
            )

        push_status_line = "disabled (enable_track_push:=false)"
        if self.enable_track_push:
            self._track_queue = queue.Queue(maxsize=1)
            self._push_stop = threading.Event()
            self._track_sub = self.create_subscription(
                TrackArrayMsg, self.track_array_topic, self._track_array_callback, 10
            )
            self._tx_thread = threading.Thread(target=self._push_loop, daemon=True)
            self._tx_thread.start()
            push_status_line = f"{self.track_array_topic} -> TrackService/PushTracks"

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║       gRPC LiDAR Client (receiver)       ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Pulling from  : {self._target}  (LidarService/StreamScan)\n"
            f"  Republishing  : {self.scan_topic}  (frame '{self.frame_id}')\n"
            f"  Timestamps    : {'source (robot clock)' if self.use_source_timestamp else 'local arrival time'}\n"
            f"  Odometry      : {odom_status_line}\n"
            f"  Pushing tracks: {push_status_line}\n"
            f"  Set 'server_address' to the host running lidar_grpc_bridge."
        )

    def _receive_loop(self):
        """Blocking pull loop with reconnect. Runs on its own thread."""
        max_bytes = self.max_message_mb * 1024 * 1024
        options = [
            ("grpc.max_receive_message_length", max_bytes),
            ("grpc.max_send_message_length", max_bytes),
        ]

        while not self._stop.is_set() and rclpy.ok():
            channel = None
            try:
                channel = grpc.insecure_channel(self._target, options=options)
                stub = lidar_stream_pb2_grpc.LidarServiceStub(channel)
                self.get_logger().info(f"Connecting to {self._target} ...")

                for scan_pb in stub.StreamScan(lidar_stream_pb2.StreamRequest()):
                    if self._stop.is_set() or not rclpy.ok():
                        break
                    if not self._connected:
                        self._connected = True
                        self.get_logger().info(f"Connected — receiving scans from {self._target}")
                    self._publish_scan(scan_pb)

            except grpc.RpcError as e:
                # UNAVAILABLE just means the bridge is not up yet; keep retrying quietly
                # rather than treating first-start ordering as a fatal error.
                code = e.code() if hasattr(e, "code") else None
                self.get_logger().warn(f"gRPC stream ended ({code}); retrying in {self.reconnect_delay_s}s")
            except Exception as e:
                self.get_logger().error(f"Unexpected error in receive loop: {e}")
            finally:
                self._connected = False
                if channel is not None:
                    channel.close()

            if self._stop.is_set() or not rclpy.ok():
                break
            time.sleep(self.reconnect_delay_s)

    def _publish_scan(self, scan_pb):
        msg = LaserScan()

        if self.use_source_timestamp and scan_pb.timestamp > 0.0:
            sec = int(scan_pb.timestamp)
            nanosec = int(round((scan_pb.timestamp - sec) * 1e9))
            if nanosec >= 1_000_000_000:   # rounding can land exactly on the next second
                sec += 1
                nanosec -= 1_000_000_000
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
        else:
            msg.header.stamp = self.get_clock().now().to_msg()

        # The proto carries no frame_id (or intensities) — the consumer supplies the
        # frame, which must match whatever TF this machine has for the lidar.
        msg.header.frame_id = self.frame_id
        msg.angle_min = scan_pb.angle_min
        msg.angle_max = scan_pb.angle_max
        msg.angle_increment = scan_pb.angle_increment
        msg.range_min = scan_pb.range_min
        msg.range_max = scan_pb.range_max
        msg.ranges = list(scan_pb.ranges)

        self._scan_pub.publish(msg)
        self._scan_count += 1
        self._scans_since_log += 1

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        state = "connected" if self._connected else "disconnected"
        line = f"[gRPC client] {state} | {rate:.1f} scans/s out | {self._scan_count} total"
        if self.enable_odom:
            odom_rate = self._odom_since_log / self.status_log_period_s
            self._odom_since_log = 0
            odom_state = "connected" if self._odom_connected else "disconnected"
            line += f"   ||   odom {odom_state} | {odom_rate:.1f} odom/s in | {self._odom_count} total"
        self.get_logger().info(line)

    # ── Odometry: pulled over gRPC, republished as /odom + tf(odom->base_link) ──────
    # Mirrors what kobuki_base_node publishes locally on the robot (see
    # botzilla_control/kobuki_base_node.py) so dr_spaam_tracker_node's
    # lookup_transform(target_frame="odom", ...) and RViz see the same TF either way,
    # without needing native ROS2 DDS discovery between the two machines.

    def _odom_receive_loop(self):
        """Blocking pull loop with reconnect, independent of the scan pull loop so a
        stall on one stream cannot block the other. Runs on its own thread."""
        while not self._odom_stop.is_set() and rclpy.ok():
            channel = None
            try:
                channel = grpc.insecure_channel(self._target)
                stub = odom_stream_pb2_grpc.OdometryServiceStub(channel)
                self.get_logger().info(f"Connecting to {self._target} for odometry ...")

                for odom_pb in stub.StreamOdometry(odom_stream_pb2.StreamRequest()):
                    if self._odom_stop.is_set() or not rclpy.ok():
                        break
                    if not self._odom_connected:
                        self._odom_connected = True
                        self.get_logger().info(f"Connected — receiving odometry from {self._target}")
                    self._publish_odom(odom_pb)

            except grpc.RpcError as e:
                code = e.code() if hasattr(e, "code") else None
                self.get_logger().warn(f"gRPC odom stream ended ({code}); retrying in {self.reconnect_delay_s}s")
            except Exception as e:
                self.get_logger().error(f"Unexpected error in odom receive loop: {e}")
            finally:
                self._odom_connected = False
                if channel is not None:
                    channel.close()

            if self._odom_stop.is_set() or not rclpy.ok():
                break
            time.sleep(self.reconnect_delay_s)

    def _publish_odom(self, odom_pb):
        if self.use_source_timestamp and odom_pb.timestamp > 0.0:
            sec = int(odom_pb.timestamp)
            nanosec = int(round((odom_pb.timestamp - sec) * 1e9))
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            stamp_sec, stamp_nanosec = sec, nanosec
        else:
            now = self.get_clock().now().to_msg()
            stamp_sec, stamp_nanosec = now.sec, now.nanosec

        msg = Odometry()
        msg.header.stamp.sec = stamp_sec
        msg.header.stamp.nanosec = stamp_nanosec
        msg.header.frame_id = self.odom_frame_id
        msg.child_frame_id = self.base_frame_id
        msg.pose.pose.position.x = odom_pb.x
        msg.pose.pose.position.y = odom_pb.y
        msg.pose.pose.orientation.z = odom_pb.qz
        msg.pose.pose.orientation.w = odom_pb.qw
        msg.twist.twist.linear.x = odom_pb.vx
        msg.twist.twist.angular.z = odom_pb.vtheta
        self._odom_pub.publish(msg)

        if self.publish_odom_tf:
            tf_msg = TransformStamped()
            tf_msg.header.stamp.sec = stamp_sec
            tf_msg.header.stamp.nanosec = stamp_nanosec
            tf_msg.header.frame_id = self.odom_frame_id
            tf_msg.child_frame_id = self.base_frame_id
            tf_msg.transform.translation.x = odom_pb.x
            tf_msg.transform.translation.y = odom_pb.y
            tf_msg.transform.rotation.z = odom_pb.qz
            tf_msg.transform.rotation.w = odom_pb.qw
            self._tf_broadcaster.sendTransform(tf_msg)

        self._odom_count += 1
        self._odom_since_log += 1

    # ── Return path: push tracks up ──────────────────────────────────────────────

    def _track_array_callback(self, msg: TrackArrayMsg):
        """ROS executor thread. Converts to protobuf here so the push thread only
        ever handles already-built messages, matching how _scan_callback works on
        the server side of this same link."""
        stamp = msg.header.stamp
        pb = track_stream_pb2.TrackArray(
            timestamp=stamp.sec + stamp.nanosec * 1e-9,
            frame_id=msg.header.frame_id,
        )
        for t in msg.tracks:
            pb.tracks.add(
                id=t.id, x=t.x, y=t.y, vx=t.vx, vy=t.vy,
                speed=t.speed, heading=t.heading, moving=t.moving,
            )
        try:
            self._track_queue.put_nowait(pb)
        except queue.Full:
            # Latest-wins: a stale track set is worse for a reacting planner than one
            # that arrives a beat late but current, same reasoning as the scan side.
            try:
                self._track_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._track_queue.put_nowait(pb)
            except queue.Full:
                pass

    def _track_generator(self):
        """Consumed by stub.PushTracks(...). Blocks between items rather than
        polling; returning ends the RPC cleanly so the call can be retried."""
        while not self._push_stop.is_set() and rclpy.ok():
            try:
                yield self._track_queue.get(timeout=0.5)
            except queue.Empty:
                continue

    def _push_loop(self):
        """Independent reconnect loop for the push direction — see module docstring
        for why this is a separate connection rather than sharing the pull channel."""
        while not self._push_stop.is_set() and rclpy.ok():
            channel = None
            try:
                channel = grpc.insecure_channel(self._target)
                stub = track_stream_pb2_grpc.TrackServiceStub(channel)
                ack = stub.PushTracks(self._track_generator())
                self.get_logger().info(f"Track push stream ended (acked {ack.tracks_received} track(s))")
            except grpc.RpcError as e:
                code = e.code() if hasattr(e, "code") else None
                self.get_logger().warn(f"Track push failed ({code}); retrying in {self.reconnect_delay_s}s")
            except Exception as e:
                self.get_logger().error(f"Unexpected error in track push loop: {e}")
            finally:
                if channel is not None:
                    channel.close()

            if self._push_stop.is_set() or not rclpy.ok():
                break
            time.sleep(self.reconnect_delay_s)

    def shutdown(self):
        self._stop.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=3.0)
        if self.enable_odom:
            self._odom_stop.set()
            if self._odom_rx_thread.is_alive():
                self._odom_rx_thread.join(timeout=3.0)
        if self.enable_track_push:
            self._push_stop.set()
            if self._tx_thread.is_alive():
                self._tx_thread.join(timeout=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GrpcScanClientNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting gRPC scan client node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
