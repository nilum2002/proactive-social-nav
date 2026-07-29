"""grpc_lidar_server_node.py.

Bridges ROS2 sensor_msgs/LaserScan onto the gRPC LidarService defined in
proto/lidar_stream.proto, so an off-board machine (e.g. the inference laptop at
192.168.0.200) can pull the live scan stream without needing ROS2 on the network.

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

import grpc
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# The generated stubs use absolute imports (`import lidar_stream_pb2`) because that is
# what protoc emits, and jetson_grpc_client.py imports them the same way. Rather than
# hand-editing generated code — which would make every regeneration a conflict — put
# this package's own directory on sys.path so those absolute imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_stream_pb2          # noqa: E402
import lidar_stream_pb2_grpc     # noqa: E402


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


class LidarGrpcServerNode(Node):

    def __init__(self):
        super().__init__("lidar_grpc_server_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("bind_address", "0.0.0.0")
        self.declare_parameter("port", 50051)
        self.declare_parameter("max_clients", 4)
        self.declare_parameter("max_message_mb", 10)
        self.declare_parameter("status_log_period_s", 5.0)

        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.bind_address = self.get_parameter("bind_address").get_parameter_value().string_value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.max_clients = self.get_parameter("max_clients").get_parameter_value().integer_value
        self.max_message_mb = self.get_parameter("max_message_mb").get_parameter_value().integer_value
        self.status_log_period_s = self.get_parameter("status_log_period_s").get_parameter_value().double_value

        self._scan_count = 0
        self._scans_since_log = 0

        self._servicer = LidarServicer(self.get_logger())

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

        if self.status_log_period_s > 0.0:
            self._status_timer = self.create_timer(self.status_log_period_s, self._log_status)

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║        LiDAR → gRPC Stream Bridge        ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Subscribed to : {self.scan_topic}\n"
            f"  Serving on    : {bind_target}  (LidarService/StreamScan)\n"
            f"  Max clients   : {self.max_clients}\n"
            f"  Max message   : {self.max_message_mb} MiB\n"
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

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        self.get_logger().info(
            f"[gRPC] {self._servicer.client_count} client(s) | "
            f"{rate:.1f} scans/s in | {self._scan_count} total"
        )

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
