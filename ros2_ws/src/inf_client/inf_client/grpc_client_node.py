"""inf_client: forwards /scan and /odom to inf_server over gRPC.

The robot is the gRPC CLIENT (reverse of lidar_grpc_bridge): it dials into
inf_server and pushes an interleaved stream of SensorFrame messages built from
its own /scan and /odom topics. inf_server runs DR-SPAAM + Kalman tracking on
the other end; nothing is expected back on this call (client-streaming), so
this node only uploads.
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

# protoc emits absolute imports (`import perception_stream_pb2`); put this
# package's own directory on sys.path so those resolve without hand-editing the
# generated code, matching inf_server's convention.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception_stream_pb2          # noqa: E402
import perception_stream_pb2_grpc     # noqa: E402


class InfClientNode(Node):

    def __init__(self):
        super().__init__("inf_client_node")

        self.declare_parameter("server_address", "192.168.1.100:50053")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("max_message_mb", 10)
        self.declare_parameter("queue_size", 20)
        self.declare_parameter("reconnect_delay_s", 2.0)
        self.declare_parameter("status_log_period_s", 5.0)

        gp = self.get_parameter
        self.server_address = gp("server_address").get_parameter_value().string_value
        self.scan_topic = gp("scan_topic").get_parameter_value().string_value
        self.odom_topic = gp("odom_topic").get_parameter_value().string_value
        self.max_message_mb = gp("max_message_mb").get_parameter_value().integer_value
        self.queue_size = gp("queue_size").get_parameter_value().integer_value
        self.reconnect_delay_s = gp("reconnect_delay_s").get_parameter_value().double_value
        self.status_log_period_s = gp("status_log_period_s").get_parameter_value().double_value

        # Bounded + drop-oldest rather than unbounded: if the link to inf_server
        # is slower than the sensors publish, old frames are worse than useless
        # (they'd just add lag), so the freshest one always wins a full queue.
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._frames_sent = 0
        self._frames_sent_since_log = 0
        self._dropped = 0

        self._scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self._odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._stream_loop, daemon=True)
        self._worker.start()

        if self.status_log_period_s > 0.0:
            self.create_timer(self.status_log_period_s, self._log_status)

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║   inf_client: /scan + /odom -> inf_server ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Target server : {self.server_address}\n"
            f"  Forwarding    : {self.scan_topic}, {self.odom_topic}\n"
            f"  Queue size    : {self.queue_size} (drop-oldest when full)"
        )

    def _scan_callback(self, msg: LaserScan):
        stamp = msg.header.stamp
        frame = perception_stream_pb2.SensorFrame(
            scan=perception_stream_pb2.LaserScanData(
                timestamp=stamp.sec + stamp.nanosec * 1e-9,
                angle_min=msg.angle_min,
                angle_max=msg.angle_max,
                angle_increment=msg.angle_increment,
                range_min=msg.range_min,
                range_max=msg.range_max,
                # Forwarded verbatim, NaN/inf included -- the client must not
                # silently reinterpret no-return beams; that is the server's call.
                ranges=msg.ranges,
            )
        )
        self._enqueue(frame)

    def _odom_callback(self, msg: Odometry):
        stamp = msg.header.stamp
        frame = perception_stream_pb2.SensorFrame(
            odom=perception_stream_pb2.OdometryData(
                timestamp=stamp.sec + stamp.nanosec * 1e-9,
                x=msg.pose.pose.position.x,
                y=msg.pose.pose.position.y,
                qz=msg.pose.pose.orientation.z,
                qw=msg.pose.pose.orientation.w,
                vx=msg.twist.twist.linear.x,
                vtheta=msg.twist.twist.angular.z,
            )
        )
        self._enqueue(frame)

    def _enqueue(self, frame):
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass
            self._dropped += 1

    def _frame_generator(self):
        """Blocks for the next frame until the streaming call should stop."""
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            yield frame
            self._frames_sent += 1
            self._frames_sent_since_log += 1

    def _stream_loop(self):
        max_bytes = self.max_message_mb * 1024 * 1024
        options = [
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ]
        while not self._stop_event.is_set():
            try:
                with grpc.insecure_channel(self.server_address, options=options) as channel:
                    stub = perception_stream_pb2_grpc.PerceptionServiceStub(channel)
                    self.get_logger().info(f"connecting to inf_server at {self.server_address}...")
                    ack = stub.StreamSensorData(self._frame_generator())
                    self.get_logger().info(
                        f"stream closed by server (processed {ack.scans_processed} scan(s))"
                    )
            except grpc.RpcError as e:
                self.get_logger().warn(f"gRPC error talking to inf_server: {e.code()} {e.details()}")
            except Exception as e:
                self.get_logger().error(f"unexpected error in stream loop: {e}")

            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay_s)

    def _log_status(self):
        rate = self._frames_sent_since_log / self.status_log_period_s
        self._frames_sent_since_log = 0
        self.get_logger().info(
            f"[inf_client] {rate:.1f} frame(s)/s sent | {self._frames_sent} total | "
            f"{self._dropped} dropped (queue full)"
        )

    def shutdown(self):
        self.get_logger().info("Stopping inf_client...")
        self._stop_event.set()
        self._worker.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = InfClientNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting inf_client node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
