"""inf_client_udp: forwards /scan and /odom to inf_server_udp over UDP.

The UDP sibling of inf_client. Same job -- the robot pushes its own /scan and
/odom off-board so DR-SPAAM and the Kalman tracker can run on a real GPU -- but
over datagrams instead of a gRPC stream, so the two can be benchmarked against
each other on the same robot.

What changes, and why:

  * One scan is one datagram. Ranges are quantized to uint16 millimetres
    (see wire.py), which puts a 450-bin sweep at ~953 B: under the 1472 B an
    Ethernet MTU allows, so it never IP-fragments. That matters more than the
    bandwidth saving -- a fragmented scan is lost if *either* fragment is lost,
    so fragmenting would roughly double the effective scan loss rate.
  * There is no send queue. inf_client needed one because a gRPC stream applies
    backpressure; UDP has none, so a queue could only ever add latency to data
    whose entire value is being fresh. Frames go straight out from the callback
    on a non-blocking socket, and the rare send-buffer-full is counted as a drop.
  * Nothing is retransmitted. A scan that arrives 200 ms late is worse than one
    that never arrives, because it would walk the server's tracker backwards.
  * Odometry is different: it is ~50 B, so instead of retransmission it gets
    redundancy -- every packet repeats the previous few samples. Losing a pose
    then takes several consecutive drops. The server de-duplicates on timestamp.
  * UDP has no congestion control and no delivery report. The server sends a
    STATS packet back about once a second; that is both. If it reports sustained
    loss, this node decimates its own scan rate rather than keep blasting a
    link that is already dropping.
"""
import os
import random
import select
import socket
import sys
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# Support both `ros2 run` (installed package) and running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inf_client_udp import wire  # noqa: E402


class InfClientUdpNode(Node):

    def __init__(self, **kwargs):
        # **kwargs is forwarded to rclpy's Node so tests can inject
        # parameter_overrides; normal use passes nothing.
        super().__init__("inf_client_udp_node", **kwargs)

        # ── Link ────────────────────────────────────────────────────────────
        self.declare_parameter("server_address", "192.168.0.100:50054")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("robot_name", "botzilla")
        self.declare_parameter("snd_buffer_kb", 512)
        # IP DSCP for the outgoing packets. 34 (AF41) lands in the Wi-Fi video
        # access category, which gets shorter contention windows than best
        # effort and measurably cuts jitter on a busy AP. Left at 0 by default
        # because some APs and switches reclassify or police marked traffic,
        # which would be a surprising thing to opt someone into silently.
        self.declare_parameter("dscp", 0)

        # ── Session / liveness ──────────────────────────────────────────────
        self.declare_parameter("keepalive_period_s", 1.0)
        self.declare_parameter("heartbeat_idle_s", 1.0)
        self.declare_parameter("status_log_period_s", 5.0)

        # ── Odometry ────────────────────────────────────────────────────────
        self.declare_parameter("odom_redundancy", 3)
        self.declare_parameter("odom_max_rate_hz", 25.0)

        # ── Rate adaptation ─────────────────────────────────────────────────
        self.declare_parameter("adaptive_rate", True)
        self.declare_parameter("loss_high_ratio", 0.10)
        self.declare_parameter("loss_low_ratio", 0.02)
        self.declare_parameter("max_decimation", 4)

        gp = self.get_parameter
        self.server_address = gp("server_address").get_parameter_value().string_value
        self.scan_topic = gp("scan_topic").get_parameter_value().string_value
        self.odom_topic = gp("odom_topic").get_parameter_value().string_value
        self.robot_name = gp("robot_name").get_parameter_value().string_value
        self.snd_buffer_kb = gp("snd_buffer_kb").get_parameter_value().integer_value
        self.dscp = gp("dscp").get_parameter_value().integer_value

        self.keepalive_period_s = gp("keepalive_period_s").get_parameter_value().double_value
        self.heartbeat_idle_s = gp("heartbeat_idle_s").get_parameter_value().double_value
        self.status_log_period_s = gp("status_log_period_s").get_parameter_value().double_value

        self.odom_redundancy = max(1, gp("odom_redundancy").get_parameter_value().integer_value)
        self.odom_max_rate_hz = gp("odom_max_rate_hz").get_parameter_value().double_value

        self.adaptive_rate = gp("adaptive_rate").get_parameter_value().bool_value
        self.loss_high_ratio = gp("loss_high_ratio").get_parameter_value().double_value
        self.loss_low_ratio = gp("loss_low_ratio").get_parameter_value().double_value
        self.max_decimation = max(1, gp("max_decimation").get_parameter_value().integer_value)

        self._host, self._port = self._parse_address(self.server_address)
        self._odom_min_period = (1.0 / self.odom_max_rate_hz) if self.odom_max_rate_hz > 0 else 0.0

        # A fresh random id per process start. The server keys tracker state on
        # it, so a reboot resets tracking instead of resuming into a filter that
        # still holds ghosts from before the robot was power-cycled.
        self._session_id = random.getrandbits(32)

        self._scan_seq = 0
        self._odom_seq = 0
        self._odom_history = []          # oldest-first; sent reversed
        self._last_odom_sent = 0.0
        self._last_send = 0.0

        self._scans_in = 0
        self._scans_sent = 0
        self._scans_sent_since_log = 0
        self._scans_decimated = 0
        self._odom_throttled = 0
        self._packets_sent = 0
        self._bytes_sent = 0
        self._send_drops = 0             # socket send buffer full
        self._not_connected_drops = 0
        self._refused = 0                # ICMP port unreachable: nobody home
        self._last_refused_log = 0.0
        self._oversize_warned = False
        self._first_scan_logged = False

        self._decimation = 1
        self._decim_counter = 0
        self._start_mono = time.monotonic()

        # Learned from the first scans and then announced. HELLO cannot carry
        # them at startup because no scan has arrived yet.
        self._scan_bins = 0
        self._laser_frame_id = ""
        self._scan_rate_hz = 0.0
        self._last_scan_stamp = None
        self._hello_announced_geometry = False

        self._server_seen = False
        self._server_stats = None
        self._stats_lock = threading.Lock()

        self._sock = self._make_socket()
        self._connected = False
        self._ensure_connected()

        self._scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, 10)
        self._odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 10)

        self._stop = threading.Event()
        self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx_thread.start()

        # Announce before any sensor data. Waiting for the first keepalive tick
        # meant that a server which replied to our very first scan flipped
        # _server_seen before HELLO had ever gone out, so the robot stayed
        # anonymous for the whole session.
        self._send_hello()

        self.create_timer(self.keepalive_period_s, self._keepalive)
        if self.status_log_period_s > 0.0:
            self.create_timer(self.status_log_period_s, self._log_status)

        self.get_logger().info(
            f"\n"
            f"  ╔═══════════════════════════════════════════════╗\n"
            f"  ║  inf_client_udp: /scan + /odom -> inf_server   ║\n"
            f"  ╚═══════════════════════════════════════════════╝\n"
            f"  Target server : udp://{self._host}:{self._port}"
            f"  (wire v{wire.PROTO_VERSION})\n"
            f"  Session id    : {self._session_id:08x}\n"
            f"  Forwarding    : {self.scan_topic}, {self.odom_topic}\n"
            f"  Odom          : <={self.odom_max_rate_hz:.0f} Hz, "
            f"{self.odom_redundancy} sample(s)/packet\n"
            f"  Rate control  : {'adaptive' if self.adaptive_rate else 'fixed'} "
            f"(back off above {self.loss_high_ratio * 100:.0f}% reported loss)\n"
            f"  No queue, no retransmission: scans go straight out and a lost one\n"
            f"  stays lost. Freshness is the only thing worth preserving here."
        )

    @staticmethod
    def _parse_address(address):
        """Split "host:port". Raises rather than guessing a port -- silently
        defaulting would send a robot's whole sensor feed into a black hole."""
        if ":" not in address:
            raise ValueError(f"server_address must be 'host:port', got '{address}'")
        host, _, port = address.rpartition(":")
        return host, int(port)

    # ── socket ──────────────────────────────────────────────────────────────
    def _make_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        want = self.snd_buffer_kb * 1024
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, want)
        except OSError as e:
            self.get_logger().warn(f"could not set SO_SNDBUF to {want} B: {e}")
        if self.dscp:
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, (self.dscp & 0x3F) << 2)
                self.get_logger().info(f"marking packets DSCP {self.dscp}")
            except OSError as e:
                self.get_logger().warn(f"could not set DSCP {self.dscp}: {e}")
        # Fully non-blocking: a sensor callback must never stall on the network.
        # The receive side waits with select() instead.
        sock.setblocking(False)
        return sock

    def _ensure_connected(self):
        """Resolve and connect() the socket, tolerating a network that is not up yet.

        connect() on a UDP socket sends nothing; it fixes the peer so we can use
        send(), and -- the real reason -- it makes the kernel report ICMP port
        unreachable back to us. Unconnected UDP swallows that, so this is the
        difference between "the server is not running" and no feedback at all.
        """
        if self._connected:
            return True
        try:
            self._sock.connect((self._host, self._port))
        except OSError as e:
            self.get_logger().warn(f"cannot reach {self._host}:{self._port} yet ({e}); retrying")
            return False
        self._connected = True
        self.get_logger().info(f"socket connected to {self._host}:{self._port}")
        return True

    def _send(self, ptype, seq, payload):
        if not self._connected and not self._ensure_connected():
            self._not_connected_drops += 1
            return False

        data = wire.pack(ptype, self._session_id, seq, payload)
        if len(data) > wire.SAFE_DATAGRAM_BYTES and not self._oversize_warned:
            self._oversize_warned = True
            self.get_logger().error(
                f"{wire.TYPE_NAMES.get(ptype, ptype)} datagram is {len(data)} B, over the "
                f"{wire.SAFE_DATAGRAM_BYTES} B budget -- it will IP-fragment, and a fragmented "
                f"scan is lost if either half is. Reduce angle_bins or raise the MTU."
            )

        try:
            self._sock.send(data)
        except BlockingIOError:
            # Send buffer full: the link cannot absorb what the sensors produce.
            self._send_drops += 1
            return False
        except ConnectionRefusedError:
            # Asynchronous ICMP from an earlier send -- nothing is listening.
            self._refused += 1
            now = time.monotonic()
            if now - self._last_refused_log >= 5.0:
                self._last_refused_log = now
                self.get_logger().warn(
                    f"{self._host}:{self._port} refused our packets "
                    f"({self._refused}x) -- is inf_server_udp running?"
                )
            return False
        except OSError as e:
            self._send_drops += 1
            self.get_logger().warn(f"send failed: {e}")
            return False

        self._packets_sent += 1
        self._bytes_sent += len(data)
        self._last_send = time.monotonic()
        return True

    # ── ROS callbacks ───────────────────────────────────────────────────────
    def _scan_callback(self, msg):
        self._scans_in += 1

        if self._decimation > 1:
            self._decim_counter += 1
            if self._decim_counter % self._decimation:
                self._scans_decimated += 1
                return

        stamp = msg.header.stamp
        pkt = wire.pb.ScanPacket(
            timestamp=stamp.sec + stamp.nanosec * 1e-9,
            angle_min=msg.angle_min,
            angle_max=msg.angle_max,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max,
            # NaN/inf collapse to the no-return sentinel, which is the only
            # reinterpretation this node does: it says "this beam gave nothing",
            # and it stays the server's call what that means for detection.
            ranges_mm=wire.quantize_ranges(msg.ranges),
        )
        self._scan_seq += 1
        if self._send(wire.T_SCAN, self._scan_seq, pkt):
            self._scans_sent += 1
            self._scans_sent_since_log += 1

        self._scan_bins = len(msg.ranges)
        self._laser_frame_id = msg.header.frame_id
        now_stamp = pkt.timestamp
        if self._last_scan_stamp is not None:
            period = now_stamp - self._last_scan_stamp
            if 0.0 < period <= 2.0:
                rate = 1.0 / period
                self._scan_rate_hz = rate if not self._scan_rate_hz \
                    else 0.2 * rate + 0.8 * self._scan_rate_hz
        self._last_scan_stamp = now_stamp

        if not self._first_scan_logged:
            self._first_scan_logged = True
            size = len(wire.pack(wire.T_SCAN, self._session_id, self._scan_seq, pkt))
            self.get_logger().info(
                f"first scan: {len(msg.ranges)} bins -> {size} B datagram "
                f"({'fits' if size <= wire.SAFE_DATAGRAM_BYTES else 'OVER'} the "
                f"{wire.SAFE_DATAGRAM_BYTES} B budget)"
            )
        elif not self._hello_announced_geometry and self._scan_rate_hz:
            # The startup HELLO could not carry bins/frame/rate. Re-announce now
            # that they are known, so the server can sanity-check scan sizes.
            self._hello_announced_geometry = True
            self._send_hello()

    def _odom_callback(self, msg):
        now = time.monotonic()
        if self._odom_min_period and (now - self._last_odom_sent) < self._odom_min_period:
            # The server pairs odom to scans by nearest timestamp within a
            # tolerance, so pushing poses far faster than that only costs
            # bandwidth on a link the scans need.
            self._odom_throttled += 1
            return
        self._last_odom_sent = now

        stamp = msg.header.stamp
        sample = wire.pb.OdomSample(
            timestamp=stamp.sec + stamp.nanosec * 1e-9,
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            qz=msg.pose.pose.orientation.z,
            qw=msg.pose.pose.orientation.w,
            vx=msg.twist.twist.linear.x,
            vtheta=msg.twist.twist.angular.z,
        )
        self._odom_history.append(sample)
        del self._odom_history[:-self.odom_redundancy]

        # Newest first, then the redundancy copies of poses already sent.
        pkt = wire.pb.OdomPacket(samples=list(reversed(self._odom_history)))
        self._odom_seq += 1
        self._send(wire.T_ODOM, self._odom_seq, pkt)

    # ── session upkeep ──────────────────────────────────────────────────────
    def _send_hello(self):
        return self._send(wire.T_HELLO, 0, wire.pb.HelloPacket(
            robot_name=self.robot_name,
            scan_bins=self._scan_bins,
            scan_rate_hz=self._scan_rate_hz,
            laser_frame_id=self._laser_frame_id,
        ))

    def _keepalive(self):
        """Re-announce until the server answers, then just prove we are alive.

        HELLO is a datagram like any other and can be lost, so it is repeated
        rather than sent once and assumed delivered. The server does not require
        it -- it opens a session on any packet -- but without it the robot shows
        up nameless.
        """
        if not self._ensure_connected():
            return
        if not self._server_seen:
            self._send_hello()
            return
        if time.monotonic() - self._last_send > self.heartbeat_idle_s:
            self._send(wire.T_HEARTBEAT, 0, wire.pb.HeartbeatPacket(timestamp=time.time()))

    def _recv_loop(self):
        """Listen for the server's STATS: UDP's missing delivery report."""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._sock], [], [], 0.5)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                data = self._sock.recv(wire.RECV_BUFFER_BYTES)
            except (BlockingIOError, ConnectionRefusedError):
                continue
            except OSError:
                if not self._stop.is_set():
                    continue
                break

            try:
                ver, ptype, _flags, sid, _seq, _send_us, body = wire.unpack_header(data)
            except wire.WireError:
                continue
            if ver != wire.PROTO_VERSION:
                self.get_logger().warn(
                    f"server speaks wire v{ver}, we speak v{wire.PROTO_VERSION} -- "
                    f"regenerate both ends from the same perception_udp.proto"
                )
                continue
            if ptype != wire.T_STATS or sid != self._session_id:
                continue

            try:
                stats = wire.decode_payload(ptype, body)
            except Exception:
                continue

            if not self._server_seen:
                self._server_seen = True
                self.get_logger().info("server acknowledged our session")
            with self._stats_lock:
                self._server_stats = stats
            self._adapt(stats)

    def _adapt(self, stats):
        """Back off when the server says packets are not getting through.

        UDP will happily let us saturate a link and keep saturating it. This is
        the only brake in the system, so it is deliberately gentle: one step per
        report, with a gap between the back-off and recovery thresholds so it
        settles instead of oscillating around a single number.
        """
        if not self.adaptive_rate:
            return
        old = self._decimation
        pressure = max(stats.loss_ratio, 0.0)
        server_busy = stats.scans_dropped_busy > stats.scans_received * 0.2
        if pressure > self.loss_high_ratio or server_busy:
            self._decimation = min(self.max_decimation, self._decimation + 1)
        elif pressure < self.loss_low_ratio:
            self._decimation = max(1, self._decimation - 1)
        if self._decimation != old:
            self.get_logger().warn(
                f"rate adapt: sending every {self._decimation} scan(s) "
                f"(was {old}); server reports {pressure * 100:.1f}% loss, "
                f"{stats.scans_dropped_busy} dropped while busy"
            )

    # ── reporting ───────────────────────────────────────────────────────────
    def _log_status(self):
        rate = self._scans_sent_since_log / self.status_log_period_s
        self._scans_sent_since_log = 0
        elapsed = max(1e-6, time.monotonic() - self._start_mono)
        kbps = (self._bytes_sent * 8.0 / 1000.0) / elapsed

        line = (
            f"[inf_client_udp] {rate:.1f} scan(s)/s sent | "
            f"{self._scans_sent}/{self._scans_in} forwarded"
        )
        if self._decimation > 1:
            line += f" (1-in-{self._decimation}, {self._scans_decimated} skipped)"
        if self._send_drops or self._not_connected_drops:
            line += (f" | {self._send_drops} buffer-full, "
                     f"{self._not_connected_drops} unsent")
        line += f" | {kbps:.0f} kbit/s avg"
        self.get_logger().info(line)

        with self._stats_lock:
            stats = self._server_stats
        if stats is None:
            self.get_logger().info("    server: no STATS yet (is inf_server_udp up?)")
        else:
            self.get_logger().info(
                f"    server: {stats.scans_received} received, {stats.scans_lost} lost "
                f"({stats.loss_ratio * 100:.1f}%), {stats.scans_reordered} late, "
                f"{stats.scans_dropped_busy} dropped busy | jitter {stats.jitter_ms:.1f} ms | "
                f"{stats.inference_fps:.1f} FPS | {stats.active_tracks} track(s)"
            )

    def shutdown(self):
        self.get_logger().info("Stopping inf_client_udp...")
        # Courtesy only: this may be lost, and a power-cut robot never sends it
        # at all, so the server's timeout is the real teardown path.
        self._send(wire.T_BYE, 0, wire.pb.ByePacket(reason="node shutting down"))
        self._stop.set()
        self._rx_thread.join(timeout=2.0)
        try:
            self._sock.close()
        except OSError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = InfClientUdpNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        print(f"Error starting inf_client_udp node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
