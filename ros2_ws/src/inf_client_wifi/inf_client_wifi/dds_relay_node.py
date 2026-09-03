"""inf_client_wifi: forwards /scan and /odom off-board over plain ROS 2 DDS.

The third sibling of inf_client (gRPC over TCP) and inf_client_udp (raw
datagrams), and the one with no custom transport at all: DDS carries the real
sensor_msgs/LaserScan and nav_msgs/Odometry across the WiFi link, and
inf_server subscribes to them like any other ROS topic. This is the control
condition -- "why not just use ROS 2?" -- that the other two are measured
against.

Why a relay node exists at all, when DDS could carry /scan directly:

  * QoS. The drivers publish /scan RELIABLE, which is the single worst setting
    for a lossy WiFi link: DDS will retransmit a scan that is already stale,
    and under sustained loss those retransmissions crowd out fresh samples.
    This node re-publishes under BEST_EFFORT + KEEP_LAST(1), so a lost scan
    stays lost and the next one is current. That mirrors what inf_client_udp
    gets for free from UDP, which is what makes the comparison fair.
  * Blast radius. Only this node's participant is opened to the subnet; the
    driver stack keeps its localhost-only discovery. So exactly one process
    talks off-board, the same as the other two clients.
  * Instrumentation. The siblings report scans/s, forwarded/dropped and
    bandwidth. Without a node in the path there is nothing to report from, and
    three transports with two sets of numbers cannot be compared.

Discovery is NOT configured here -- it is process environment, so it lives in
wifi_client.sh. This node only logs what it found in effect, so a captured run
says how it was discovered as well as how it performed.

Known asymmetry when reading the numbers against inf_client_udp: a LaserScan
carries float32 ranges (4 B/bin), where the UDP wire format quantizes to uint16
millimetres (2 B/bin). At 500 bins that is ~2 kB versus ~1 kB on the wire before
either one has lost anything. Some of any bandwidth gap is that, not transport.
"""
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


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
    """QoS for one endpoint.

    KEEP_LAST with a shallow depth is deliberate: a queue of scans is a queue of
    stale robot poses, and delivering those late is worse for the tracker than
    never delivering them. VOLATILE for the same reason -- a subscriber that
    joins late wants the next scan, not the last one.

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


class InfClientWifiNode(Node):

    def __init__(self, **kwargs):
        # **kwargs is forwarded to rclpy's Node so tests can inject
        # parameter_overrides; normal use passes nothing.
        super().__init__("inf_client_wifi_node", **kwargs)

        # ── Topics ──────────────────────────────────────────────────────────
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("out_scan_topic", "/wifi/scan")
        self.declare_parameter("out_odom_topic", "/wifi/odom")
        self.declare_parameter("robot_name", "botzilla")

        # ── QoS ─────────────────────────────────────────────────────────────
        # Input is BEST_EFFORT because a BEST_EFFORT subscriber is compatible
        # with both RELIABLE and BEST_EFFORT publishers, while a RELIABLE
        # subscriber silently receives nothing from a BEST_EFFORT one. The
        # drivers are local, so there is nothing to gain from reliability here.
        self.declare_parameter("input_reliability", "best_effort")
        self.declare_parameter("output_reliability", "best_effort")
        self.declare_parameter("output_depth", 1)

        # ── Rate ────────────────────────────────────────────────────────────
        self.declare_parameter("scan_decimation", 1)
        self.declare_parameter("odom_max_rate_hz", 25.0)
        self.declare_parameter("status_log_period_s", 5.0)

        gp = self.get_parameter
        self.scan_topic = gp("scan_topic").get_parameter_value().string_value
        self.odom_topic = gp("odom_topic").get_parameter_value().string_value
        self.out_scan_topic = gp("out_scan_topic").get_parameter_value().string_value
        self.out_odom_topic = gp("out_odom_topic").get_parameter_value().string_value
        self.robot_name = gp("robot_name").get_parameter_value().string_value

        self.input_reliability = gp("input_reliability").get_parameter_value().string_value
        self.output_reliability = gp("output_reliability").get_parameter_value().string_value
        self.output_depth = gp("output_depth").get_parameter_value().integer_value

        self.scan_decimation = max(1, gp("scan_decimation").get_parameter_value().integer_value)
        self.odom_max_rate_hz = gp("odom_max_rate_hz").get_parameter_value().double_value
        self.status_log_period_s = gp("status_log_period_s").get_parameter_value().double_value

        # ── Counters ────────────────────────────────────────────────────────
        self._scans_in = 0
        self._scans_out = 0
        self._scans_out_since_log = 0
        self._scans_decimated = 0
        self._odom_in = 0
        self._odom_out = 0
        self._odom_throttled = 0
        self._bytes_out = 0
        self._decim_counter = 0
        self._last_odom_out = 0.0
        self._start_mono = time.monotonic()

        # Serializing every message only to weigh it would burn CPU on the Pi
        # for a number that does not change while the scan geometry holds. Size
        # is measured once and re-measured only when the bin count moves.
        self._scan_bins = 0
        self._scan_bytes = 0
        self._odom_bytes = 0
        self._first_scan_logged = False

        in_qos = build_qos(self.input_reliability, 5)
        out_qos = build_qos(self.output_reliability, self.output_depth)

        self._scan_pub = self.create_publisher(LaserScan, self.out_scan_topic, out_qos)
        self._odom_pub = self.create_publisher(Odometry, self.out_odom_topic, out_qos)
        self._scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, in_qos)
        self._odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, in_qos)

        self._status_timer = self.create_timer(self.status_log_period_s, self._log_status)
        self._log_banner()

    # ── forwarding ──────────────────────────────────────────────────────────
    def _scan_callback(self, msg: LaserScan):
        self._scans_in += 1

        if self.scan_decimation > 1:
            self._decim_counter += 1
            if self._decim_counter % self.scan_decimation:
                self._scans_decimated += 1
                return

        if len(msg.ranges) != self._scan_bins:
            self._scan_bins = len(msg.ranges)
            self._scan_bytes = len(serialize_message(msg))
            if not self._first_scan_logged:
                self._first_scan_logged = True
                self.get_logger().info(
                    f"first scan: {self._scan_bins} bins -> {self._scan_bytes} B serialized "
                    f"(CDR, before RTPS and UDP/IP headers)"
                )
            else:
                self.get_logger().info(
                    f"scan geometry changed: {self._scan_bins} bins -> {self._scan_bytes} B"
                )

        self._scan_pub.publish(msg)
        self._scans_out += 1
        self._scans_out_since_log += 1
        self._bytes_out += self._scan_bytes

    def _odom_callback(self, msg: Odometry):
        self._odom_in += 1

        # The server pairs odom to scans by nearest timestamp, so anything past
        # this rate is bandwidth spent where the scans need it. Matches
        # inf_client_udp's odom_max_rate_hz so the two carry the same odom load.
        if self.odom_max_rate_hz > 0.0:
            now = time.monotonic()
            if now - self._last_odom_out < 1.0 / self.odom_max_rate_hz:
                self._odom_throttled += 1
                return
            self._last_odom_out = now

        if not self._odom_bytes:
            self._odom_bytes = len(serialize_message(msg))

        self._odom_pub.publish(msg)
        self._odom_out += 1
        self._bytes_out += self._odom_bytes

    # ── reporting ───────────────────────────────────────────────────────────
    def _log_status(self):
        rate = self._scans_out_since_log / self.status_log_period_s
        self._scans_out_since_log = 0
        elapsed = max(1e-6, time.monotonic() - self._start_mono)
        kbps = (self._bytes_out * 8.0 / 1000.0) / elapsed

        line = (
            f"[inf_client_wifi] {rate:.1f} scan(s)/s published | "
            f"{self._scans_out}/{self._scans_in} forwarded"
        )
        if self.scan_decimation > 1:
            line += f" (1-in-{self.scan_decimation}, {self._scans_decimated} skipped)"
        line += f" | {kbps:.0f} kbit/s avg"
        self.get_logger().info(line)

        # DDS gives no delivery report to the publisher under BEST_EFFORT, so
        # unlike inf_client_udp there is no server-side loss figure to print
        # here. Subscriber count is the one thing this side can actually see:
        # zero means discovery has not crossed the link, which is by far the
        # most common way this setup fails.
        subs = self._scan_pub.get_subscription_count()
        if subs == 0:
            self.get_logger().warn(
                f"    no subscriber on {self.out_scan_topic} -- inf_server has not been "
                f"discovered (check ROS_STATIC_PEERS / ROS_DOMAIN_ID and the firewall)"
            )
        else:
            self.get_logger().info(
                f"    {subs} subscriber(s) on {self.out_scan_topic} | "
                f"odom {self._odom_out}/{self._odom_in} forwarded, "
                f"{self._odom_throttled} throttled"
            )

    def _log_banner(self):
        env = "\n".join(
            f"  {k:<32}{os.environ.get(k, '<unset>')}" for k in DISCOVERY_ENV
        )
        self.get_logger().info(
            f"\n"
            f"  ╔═══════════════════════════════════════════════╗\n"
            f"  ║  inf_client_wifi: /scan + /odom -> plain DDS   ║\n"
            f"  ╚═══════════════════════════════════════════════╝\n"
            f"  Robot         : {self.robot_name}\n"
            f"  In            : {self.scan_topic}, {self.odom_topic} "
            f"({self.input_reliability})\n"
            f"  Out           : {self.out_scan_topic}, {self.out_odom_topic} "
            f"({self.output_reliability}, keep-last {self.output_depth})\n"
            f"  Scan rate     : 1-in-{self.scan_decimation}\n"
            f"  Odom          : <={self.odom_max_rate_hz:g} Hz\n"
            f"  Discovery (process environment, not set by this node):\n"
            f"{env}\n"
            f"  No queue of our own and no retransmission: BEST_EFFORT means a\n"
            f"  lost scan stays lost, so the next one is always the current one."
        )

    def shutdown(self):
        self.get_logger().info(
            f"Stopping inf_client_wifi... {self._scans_out}/{self._scans_in} scans, "
            f"{self._odom_out}/{self._odom_in} odom forwarded"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = InfClientWifiNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        print(f"Error starting inf_client_wifi node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
