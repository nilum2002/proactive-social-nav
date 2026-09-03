"""inf_server_udp: DR-SPAAM + Kalman tracking over UDP datagrams.

The UDP sibling of inf_server. Same job -- receive a robot's /scan and /odom,
run the detector and tracker off-board, publish tracks and TF locally -- but
over datagrams instead of a gRPC stream, so the two transports can be compared
on the same robot and the same GPU.

What UDP changes, and how this node answers each:

  * No connection. A session is whatever arrives carrying a session_id, and it
    ends when packets stop. `Session` holds the per-robot tracker, so two robots
    never share filter state, and a robot that reboots (new random session_id)
    gets a clean tracker instead of resuming into one still holding ghosts.

  * No backpressure. gRPC blocked the sender when the detector fell behind; UDP
    cannot. If a scan arrives while the previous one is still in the detector,
    the *older* one is discarded and counted as `scans_dropped_busy`. Keeping
    the newer scan is the whole point: a stale scan walks the tracker backwards.

  * No delivery report. Sequence numbers are per packet type, so a gap in the
    scan stream is measured without odom traffic masking it. Losses, reorders
    and RFC 3550 interarrival jitter are accumulated per session and mailed back
    once a second as a StatsPacket -- which is what the client's rate adaptation
    uses as its only congestion signal.

  * No delivery order. A scan with a sequence number at or below the newest one
    already seen is dropped rather than buffered, for the same reason a late
    scan is worthless: it would rewind the tracker's state.

  * An open port receives whatever the internet sends it. Every datagram is
    validated -- length, magic, version -- before any protobuf decode, because a
    version-mismatched protobuf usually parses into plausible garbage rather
    than raising. Malformed traffic is counted, not fatal.
"""
import math
import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped, Vector3
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster

from dr_spaam.detector import Detector

from inf_server_udp import wire
from inf_server_udp.kalman_tracker import MultiObjectTracker


class ScanView:
    """A decoded ScanPacket shaped like the gRPC path's scan message.

    `_detect` and `republish_scan` are shared with inf_server and expect
    `.ranges` in metres plus the geometry fields; this adapts the quantized
    wire form to that without either side knowing about the other.
    """

    __slots__ = ("timestamp", "angle_min", "angle_max", "angle_increment",
                 "range_min", "range_max", "ranges", "seq", "recv_mono")

    def __init__(self, pkt, seq, recv_mono):
        self.timestamp = pkt.timestamp
        self.angle_min = pkt.angle_min
        self.angle_max = pkt.angle_max
        self.angle_increment = pkt.angle_increment
        self.range_min = pkt.range_min
        self.range_max = pkt.range_max
        # inf for a no-return beam, exactly like a driver's LaserScan, so the
        # detector's existing inf handling applies unchanged.
        self.ranges = wire.dequantize_ranges(pkt.ranges_mm)
        self.seq = seq
        self.recv_mono = recv_mono


class SeqTracker:
    """Loss / reorder accounting for one packet-type stream.

    Sequence numbers are per type, so scan loss is not diluted by odom packets
    arriving at a different rate. Counting is gap-based: it cannot distinguish a
    dropped packet from one still in flight, which is the correct trade for a
    signal whose only job is to tell the sender to slow down.
    """

    def __init__(self):
        self.received = 0
        self.lost = 0
        self.reordered = 0
        self.highest_seq = None

    def observe(self, seq):
        """Returns True if this packet should be processed."""
        self.received += 1
        if self.highest_seq is None:
            self.highest_seq = seq
            return True
        if seq <= self.highest_seq:
            # Late or duplicate. Never buffered: a scan that arrives after a
            # newer one has nothing left to contribute but a backwards step.
            self.reordered += 1
            # It was counted as lost when the gap opened, but it did arrive.
            # Un-counting matters: loss_ratio is the client's only congestion
            # signal, and a link that merely reorders would otherwise look like
            # a lossy one and make the robot throttle its scan rate for nothing.
            if self.lost > 0:
                self.lost -= 1
            return False
        self.lost += seq - self.highest_seq - 1
        self.highest_seq = seq
        return True

    @property
    def loss_ratio(self):
        expected = self.received + self.lost
        return (self.lost / expected) if expected else 0.0


class Session:
    """Everything the server knows about one robot, keyed on session_id."""

    # RFC 3550 interarrival jitter gain. 1/16 is the value the RTP spec
    # specifies; it is slow enough to ignore single outliers.
    JITTER_GAIN = 1.0 / 16.0

    def __init__(self, session_id, addr, tracker_kwargs, odom_history):
        self.session_id = session_id
        self.addr = addr
        self.robot_name = f"{session_id:08x}"
        self.scan_bins = 0
        self.scan_rate_hz = 0.0
        self.laser_frame_id = ""

        self.tracker = MultiObjectTracker(**tracker_kwargs)
        self.scan_seq = SeqTracker()
        self.odom_seq = SeqTracker()

        # Odometry is paired to scans by timestamp, so a short history is kept
        # rather than only the newest pose: the newest odom sample is often from
        # *after* the scan being processed.
        self.odom = deque(maxlen=odom_history)
        self.odom_timestamps = set()      # de-duplicates the redundant copies
        self.scans_dropped_busy = 0
        self.scans_processed = 0
        self.bad_packets = 0

        self.pending = None               # one slot: newest scan wins
        self.last_scan_time = None
        self.last_seen = time.monotonic()
        self.created = time.monotonic()

        self._jitter = 0.0
        self._last_transit_us = None

    def touch(self):
        self.last_seen = time.monotonic()

    def note_arrival(self, send_ts_us, recv_ts_us):
        """RFC 3550 interarrival jitter.

        The two clocks are not synchronised and need not be: transit is
        (recv - send) in each direction and the constant offset cancels in the
        difference between consecutive transits.
        """
        transit = recv_ts_us - send_ts_us
        if self._last_transit_us is not None:
            d = abs(transit - self._last_transit_us)
            self._jitter += (d - self._jitter) * self.JITTER_GAIN
        self._last_transit_us = transit

    @property
    def jitter_ms(self):
        return self._jitter / 1000.0

    def add_odom(self, sample):
        """Store a pose, ignoring the redundant repeats the client sends."""
        if sample.timestamp in self.odom_timestamps:
            return False
        self.odom_timestamps.add(sample.timestamp)
        if len(self.odom) == self.odom.maxlen and self.odom:
            self.odom_timestamps.discard(self.odom[0].timestamp)
        self.odom.append(sample)
        return True

    def odom_for(self, stamp, tolerance_s):
        """Pose nearest this scan's timestamp, or None if nothing is close.

        Nearest-in-time rather than most-recent: odom and scans arrive on
        independent schedules, and using a pose from 40 ms after the scan skews
        every detection by however far the robot travelled in between.
        """
        best, best_dt = None, tolerance_s
        for s in self.odom:
            dt = abs(s.timestamp - stamp)
            if dt <= best_dt:
                best, best_dt = s, dt
        if best is None:
            return None
        yaw = 2.0 * math.atan2(best.qz, best.qw)
        return (best.x, best.y, yaw)


class UdpServerNode(Node):

    def __init__(self, node_name="inf_server_udp_node"):
        super().__init__(node_name)

        # ── Detector ────────────────────────────────────────────────────────
        self.declare_parameter("weight_file", "")
        self.declare_parameter("detector_model", "DR-SPAAM")
        self.declare_parameter("conf_thresh", 0.8)
        self.declare_parameter("stride", 1)
        self.declare_parameter("panoramic_scan", True)
        self.declare_parameter("gpu", True)

        # ── UDP link ────────────────────────────────────────────────────────
        self.declare_parameter("bind_address", "0.0.0.0")
        # 50054, not 50053: the gRPC inf_server can keep running alongside.
        self.declare_parameter("port", 50054)
        self.declare_parameter("rcv_buffer_kb", 1024)
        self.declare_parameter("session_timeout_s", 5.0)
        self.declare_parameter("stats_period_s", 1.0)
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

        gp = self.get_parameter
        self.weight_file = gp("weight_file").get_parameter_value().string_value
        self.detector_model = gp("detector_model").get_parameter_value().string_value
        self.conf_thresh = gp("conf_thresh").get_parameter_value().double_value
        self.stride = gp("stride").get_parameter_value().integer_value
        self.panoramic_scan = gp("panoramic_scan").get_parameter_value().bool_value
        self.use_gpu = gp("gpu").get_parameter_value().bool_value

        self.bind_address = gp("bind_address").get_parameter_value().string_value
        self.port = gp("port").get_parameter_value().integer_value
        self.rcv_buffer_kb = gp("rcv_buffer_kb").get_parameter_value().integer_value
        self.session_timeout_s = gp("session_timeout_s").get_parameter_value().double_value
        self.stats_period_s = gp("stats_period_s").get_parameter_value().double_value
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

        if not self.weight_file:
            self.get_logger().error("Parameter 'weight_file' is empty! Provide a path to a valid checkpoint.")
            raise ValueError("weight_file parameter is empty.")

        self.get_logger().info(f"Loading detector '{self.detector_model}' from: {self.weight_file}")
        self._detector = Detector(
            self.weight_file, model=self.detector_model, gpu=self.use_gpu,
            stride=self.stride, panoramic_scan=self.panoramic_scan,
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
        self._bad_packets = 0
        self._version_mismatches = 0
        self._last_version_warn = 0.0

        self._sessions = {}
        self._sessions_lock = threading.Lock()
        self._work_cv = threading.Condition(self._sessions_lock)

        self._sock = self._make_socket()
        self._stop = threading.Event()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._workers = []
        self._start_workers()
        self._rx_thread.start()
        for w in self._workers:
            w.start()

        if self.stats_period_s > 0.0:
            self.create_timer(self.stats_period_s, self._send_all_stats)
        self.create_timer(1.0, self._reap_sessions)
        if self.status_log_period_s > 0.0:
            self.create_timer(self.status_log_period_s, self._log_status)

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║   inf_server_udp: DR-SPAAM + KF Tracker  ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Listening on : udp://{self.bind_address}:{self.port} "
            f"(wire v{wire.PROTO_VERSION})\n"
            f"  Sessions     : keyed on session_id, reaped after "
            f"{self.session_timeout_s:g}s of silence\n"
            f"  Odom pairing : nearest sample within "
            f"{self.odom_tolerance_s * 1e3:.0f} ms of the scan\n"
            f"  Backpressure : none. A scan arriving while the detector is busy\n"
            f"  replaces the one waiting -- the newest scan is the only useful one.\n"
            f"  Stats (loss, jitter, fps) are mailed back every "
            f"{self.stats_period_s:g}s; that is\n"
            f"  the client's only congestion signal."
        )

    # ── socket / receive ────────────────────────────────────────────────────

    def _make_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        want = self.rcv_buffer_kb * 1024
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, want)
        except OSError as e:
            self.get_logger().warn(f"could not set SO_RCVBUF to {want} B: {e}")
        got = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        try:
            sock.bind((self.bind_address, self.port))
        except OSError as e:
            raise RuntimeError(
                f"failed to bind udp://{self.bind_address}:{self.port} ({e}) -- "
                f"is another inf_server_udp already running?")
        # A timeout rather than non-blocking: the receive loop has nothing else
        # to do, and the timeout is only there to notice shutdown.
        sock.settimeout(0.5)
        self.get_logger().info(
            f"bound udp://{self.bind_address}:{self.port}, SO_RCVBUF {got // 1024} KiB "
            f"(a small buffer is where scans are lost invisibly under load)")
        return sock

    def _rx_loop(self):
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(wire.RECV_BUFFER_BYTES)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            try:
                self._handle(data, addr)
            except Exception as e:      # one bad packet must not kill the loop
                self._bad_packets += 1
                self.get_logger().error(f"error handling datagram from {addr}: {e}",
                                        throttle_duration_sec=5.0)

    def _handle(self, data, addr):
        recv_us = wire.now_us()
        try:
            ver, ptype, _flags, sid, seq, send_us, body = wire.unpack_header(data)
        except wire.WireError:
            # Port scans and strays are normal on an open UDP port.
            self._bad_packets += 1
            return

        if ver != wire.PROTO_VERSION:
            self._version_mismatches += 1
            now = time.monotonic()
            if now - self._last_version_warn >= 10.0:
                self._last_version_warn = now
                self.get_logger().warn(
                    f"{addr[0]} speaks wire v{ver}, we speak v{wire.PROTO_VERSION} -- "
                    f"regenerate both ends from the same perception_udp.proto")
            return

        try:
            msg = wire.decode_payload(ptype, body)
        except Exception:
            self._bad_packets += 1
            return

        session = self._get_session(sid, addr)
        session.touch()
        session.note_arrival(send_us, recv_us)

        if ptype == wire.T_SCAN:
            self._on_scan(session, seq, msg)
        elif ptype == wire.T_ODOM:
            self._on_odom(session, seq, msg)
        elif ptype == wire.T_HELLO:
            self._on_hello(session, msg)
        elif ptype == wire.T_HEARTBEAT:
            pass                                    # touch() was the whole point
        elif ptype == wire.T_BYE:
            self._on_bye(session, msg)

    def _get_session(self, sid, addr):
        with self._sessions_lock:
            s = self._sessions.get(sid)
            if s is None:
                s = Session(sid, addr, self.tracker_kwargs, self.odom_history)
                self._sessions[sid] = s
                self.get_logger().info(
                    f"session {sid:08x} opened from {addr[0]}:{addr[1]} "
                    f"(now {len(self._sessions)} active)")
            elif s.addr != addr:
                # Same session from a new address: a robot that changed IP, e.g.
                # roamed between APs. Follow it -- the session_id is the identity
                # here, not the address.
                self.get_logger().info(
                    f"session {sid:08x} moved {s.addr[0]}:{s.addr[1]} -> {addr[0]}:{addr[1]}")
                s.addr = addr
            return s

    # ── packet handlers ─────────────────────────────────────────────────────

    def _on_hello(self, session, msg):
        session.robot_name = msg.robot_name or session.robot_name
        session.scan_bins = msg.scan_bins
        session.scan_rate_hz = msg.scan_rate_hz
        session.laser_frame_id = msg.laser_frame_id
        self.get_logger().info(
            f"session {session.session_id:08x} is '{session.robot_name}': "
            f"{msg.scan_bins} bins @ {msg.scan_rate_hz:.1f} Hz, frame "
            f"'{msg.laser_frame_id or '(unset)'}'")

    def _on_scan(self, session, seq, pkt):
        if not session.scan_seq.observe(seq):
            return                                  # late or duplicate
        view = ScanView(pkt, seq, time.monotonic())

        if session.scan_bins and len(view.ranges) != session.scan_bins:
            self.get_logger().warn(
                f"session {session.session_id:08x}: scan has {len(view.ranges)} bins, "
                f"HELLO announced {session.scan_bins}", throttle_duration_sec=10.0)

        with self._work_cv:
            if session.pending is not None:
                # The detector is still busy with the previous scan. Discard the
                # older one, not this one: freshness is the only thing worth
                # keeping, and this is what the client's rate adaptation reads.
                session.scans_dropped_busy += 1
            session.pending = view
            self._work_cv.notify()

    def _on_odom(self, session, seq, pkt):
        session.odom_seq.observe(seq)
        # Deliberately processed even when `observe` says the packet is late:
        # each OdomPacket repeats the previous few samples, so an out-of-order
        # packet can still carry a pose that was never seen. De-duplication is
        # by timestamp, which makes replaying harmless.
        newest = None
        for sample in pkt.samples:
            if session.add_odom(sample) and (newest is None or sample.timestamp > newest.timestamp):
                newest = sample
        if newest is not None:
            yaw = 2.0 * math.atan2(newest.qz, newest.qw)
            self.broadcast_odom_tf(newest.x, newest.y, yaw, newest.timestamp)

    def _on_bye(self, session, msg):
        with self._sessions_lock:
            self._sessions.pop(session.session_id, None)
        self.get_logger().info(
            f"session {session.session_id:08x} ('{session.robot_name}') said goodbye"
            f"{': ' + msg.reason if msg.reason else ''}")

    # ── work loop ───────────────────────────────────────────────────────────

    def _start_workers(self):
        """Build the processing threads. Called before any packet is accepted,
        so the pipelined variant can declare its own parameters here and swap in
        a two-stage arrangement without racing the receive loop."""
        self._workers.append(threading.Thread(target=self._worker_loop, daemon=True))

    def _worker_loop(self):
        """Detection + tracking, off the receive path.

        Sequential variant: detect and track run inline, one after the other,
        for each scan -- the arrangement grpc_server_node uses. What differs is
        only that the receive thread never waits for this one, because a UDP
        sender cannot be told to wait.
        """
        while not self._stop.is_set():
            with self._work_cv:
                item = self._take_pending()
                while item is None:
                    if self._stop.is_set():
                        return
                    self._work_cv.wait(0.2)
                    item = self._take_pending()
            session, view = item
            try:
                self.process_scan(session, view)
            except Exception as e:
                self.get_logger().error(f"scan processing failed: {e}",
                                        throttle_duration_sec=5.0)

    def _take_pending(self):
        """Claim one waiting scan. Caller holds _work_cv."""
        for session in self._sessions.values():
            if session.pending is not None:
                view, session.pending = session.pending, None
                return session, view
        return None

    def process_scan(self, session, view):
        dt = 0.1
        if session.last_scan_time is not None:
            candidate = view.timestamp - session.last_scan_time
            # A gap here is real elapsed time -- dropped scans included -- so it
            # is passed to the filter rather than clamped to one scan period.
            if 0.0 < candidate <= 2.0:
                dt = candidate
        session.last_scan_time = view.timestamp

        odom = session.odom_for(view.timestamp, self.odom_tolerance_s)
        self.republish_scan(view)

        with self._process_lock:
            dets_xy = self._detect(view)
            dets_xy, frame_id = self._to_tracking_frame(dets_xy, odom)
            active_tracks = session.tracker.step(dt, dets_xy)

        self._publish_detections_marker(dets_xy, frame_id)
        self._publish_ros(frame_id, active_tracks)

        session.scans_processed += 1
        self._scan_count += 1
        self._scans_since_log += 1

    # ── stats / liveness ────────────────────────────────────────────────────

    def _send_all_stats(self):
        """UDP's missing delivery report, mailed back once a period."""
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            pkt = wire.pb.StatsPacket(
                scans_received=s.scan_seq.received,
                scans_lost=s.scan_seq.lost,
                scans_reordered=s.scan_seq.reordered,
                loss_ratio=s.scan_seq.loss_ratio,
                jitter_ms=s.jitter_ms,
                inference_fps=self.inference_fps,
                active_tracks=len([t for t in s.tracker.tracks if t.state == "ACTIVE"])
                if hasattr(s.tracker, "tracks") else 0,
                scans_dropped_busy=s.scans_dropped_busy,
            )
            try:
                self._sock.sendto(wire.pack(wire.T_STATS, s.session_id, 0, pkt), s.addr)
            except OSError as e:
                self.get_logger().warn(
                    f"could not send STATS to {s.addr}: {e}", throttle_duration_sec=10.0)

    def _reap_sessions(self):
        """Drop sessions that have gone quiet.

        The client heartbeats while idle precisely so this can tell "parked" from
        "gone", and dropping the session frees its tracker -- otherwise a robot
        that vanished mid-run would keep publishing coasting ghosts forever.
        """
        now = time.monotonic()
        with self._sessions_lock:
            dead = [sid for sid, s in self._sessions.items()
                    if now - s.last_seen > self.session_timeout_s]
            for sid in dead:
                s = self._sessions.pop(sid)
        for sid in dead:
            self.get_logger().info(
                f"session {sid:08x} timed out after {self.session_timeout_s:g}s of silence")

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        line = (f"[inf_server_udp] {len(sessions)} session(s) | "
                f"{rate:.1f} scans/s | {self.inference_fps:.1f} FPS (DR-SPAAM) | "
                f"{self._scan_count} total")
        if self._bad_packets or self._version_mismatches:
            line += (f" | {self._bad_packets} malformed, "
                     f"{self._version_mismatches} wrong-version")
        self.get_logger().info(line)
        for s in sessions:
            self.get_logger().info(
                f"    {s.robot_name} [{s.session_id:08x}] "
                f"rx {s.scan_seq.received} lost {s.scan_seq.lost} "
                f"({s.scan_seq.loss_ratio * 100:.1f}%) reord {s.scan_seq.reordered} "
                f"busy-drop {s.scans_dropped_busy} | jitter {s.jitter_ms:.1f} ms "
                f"| odom {len(s.odom)}")

    # ── ROS output (shared with inf_server) ─────────────────────────────────

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

    def republish_scan(self, view):
        msg = LaserScan()
        sec = int(view.timestamp)
        nanosec = int(round((view.timestamp - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = self.laser_frame_id
        msg.angle_min = view.angle_min
        msg.angle_max = view.angle_max
        msg.angle_increment = view.angle_increment
        msg.range_min = view.range_min
        msg.range_max = view.range_max
        msg.ranges = [float(r) for r in view.ranges]
        self._scan_pub.publish(msg)

    def _detect(self, view):
        if not self._detector.is_ready():
            fov_rad = view.angle_increment * len(view.ranges)
            self._detector.set_laser_fov(np.rad2deg(fov_rad))
            self.get_logger().info(
                f"Dynamic LiDAR FOV configured to: {np.rad2deg(fov_rad):.2f} degrees")

        scan = np.array(view.ranges, dtype=np.float32)
        scan[scan < view.range_min] = 29.99
        scan[scan > view.range_max] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        scan_phi = view.angle_min + np.arange(len(view.ranges)) * view.angle_increment

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

    def _to_tracking_frame(self, dets_xy, odom):
        """Project detections into the odom frame so the tracker estimates
        velocity in a world-fixed frame while the robot drives.

        The frame depends only on whether a pose was matched -- never on how
        many detections this scan produced. Returning "base_link" for an empty
        scan would mislabel the frame while tracks coast on odom coordinates,
        and RViz would re-anchor them to the robot so they appear to travel
        along with it.
        """
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
        self.get_logger().info("Stopping UDP server...")
        self._stop.set()
        with self._work_cv:
            self._work_cv.notify_all()
        for w in self._workers:
            w.join(timeout=2.0)
        try:
            self._sock.close()
        except OSError:
            pass
        self._rx_thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = UdpServerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass          # Ctrl-C / SIGTERM are normal shutdowns, not errors
    except Exception as e:
        print(f"Error starting inf_server_udp node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
