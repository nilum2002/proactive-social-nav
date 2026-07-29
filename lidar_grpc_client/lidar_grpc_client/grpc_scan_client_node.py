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

Threading: the blocking gRPC receive loop owns a daemon thread; rclpy.spin() keeps the
main thread. rclpy publishers are thread-safe, so the receive thread publishes directly.
"""
import os
import sys
import threading
import time

import grpc
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Reuse the generated stubs from lidar_grpc_bridge rather than keeping a second copy
# that can drift from the .proto. They use absolute imports (`import lidar_stream_pb2`)
# because that is what protoc emits, so put their directory on sys.path to resolve it.
import lidar_grpc_bridge  # noqa: F401  (imported for its __file__ location)

sys.path.insert(0, os.path.dirname(os.path.abspath(lidar_grpc_bridge.__file__)))
import lidar_stream_pb2          # noqa: E402
import lidar_stream_pb2_grpc     # noqa: E402


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

        self.server_address = self.get_parameter("server_address").get_parameter_value().string_value
        self.server_port = self.get_parameter("server_port").get_parameter_value().integer_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.reconnect_delay_s = self.get_parameter("reconnect_delay_s").get_parameter_value().double_value
        self.max_message_mb = self.get_parameter("max_message_mb").get_parameter_value().integer_value
        self.status_log_period_s = self.get_parameter("status_log_period_s").get_parameter_value().double_value
        self.use_source_timestamp = self.get_parameter("use_source_timestamp").get_parameter_value().bool_value

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

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║       gRPC LiDAR Client (receiver)       ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Pulling from  : {self._target}  (LidarService/StreamScan)\n"
            f"  Republishing  : {self.scan_topic}  (frame '{self.frame_id}')\n"
            f"  Timestamps    : {'source (robot clock)' if self.use_source_timestamp else 'local arrival time'}\n"
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
        self.get_logger().info(
            f"[gRPC client] {state} | {rate:.1f} scans/s out | {self._scan_count} total"
        )

    def shutdown(self):
        self._stop.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=3.0)


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
