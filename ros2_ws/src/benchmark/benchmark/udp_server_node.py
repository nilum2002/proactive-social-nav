"""
What is added relative to inf_server_udp's node:

  * T_det / T_track / T_lat / T_scan / T_fwd -- the shared timing set, in
    the status log and in the CSV. T_lat is arrival (packet recv) to
    published-out, measured on perf_counter()/monotonic() within this
    process, so it is directly comparable against the gRPC and WiFi nodes'
    T_lat even though all three hand a scan over differently.

  * `loss_percent` / `jitter_ms` -- averaged across active sessions at the
    moment the row is written. inf_server_udp already computes these per
    session for its own status log; this just also lands in the CSV.

  * `busy_drop_in_window` / `bad_packets_in_window` -- windowed deltas of the
    same counters inf_server_udp already accumulates, on the same
    scans-in-window pattern used elsewhere in the CSV.

  * GPU/CPU/RAM sampling and the CSV writer itself, copied from
    grpc_server_node.py's InfServerNode.
"""
import csv
import math
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
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

from benchmark import wire
from benchmark.kalman_tracker import MultiObjectTracker
from benchmark.sys_report import (
    TIMING_FIELDNAMES, WIRE_FIELDNAMES, GpuSampler, PerfStats, stat_ms)


class ScanView:
    """A decoded ScanPacket shaped like the gRPC path's scan message.

    `_detect` and `republish_scan` expect `.ranges` in metres plus the
    geometry fields; this adapts the quantized wire form to that.
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

        self.odom = deque(maxlen=odom_history)
        self.odom_timestamps = set()
        self.scans_dropped_busy = 0
        self.scans_processed = 0
        self.bad_packets = 0

        self.pending = None
        self.last_scan_time = None
        self.last_seen = time.monotonic()
        self.created = time.monotonic()

        self._jitter = 0.0
        self._last_transit_us = None

    def touch(self):
        self.last_seen = time.monotonic()

    def note_arrival(self, send_ts_us, recv_ts_us):
        """RFC 3550 interarrival jitter. Returns the transit in seconds.

        The two clocks are not synchronised and need not be for jitter: the
        constant offset cancels in the difference between consecutive
        transits. It does NOT cancel in the transit itself, which is why the
        returned value is reported as an estimate (wire_ms) and never as a
        one-way delay.
        """
        transit = recv_ts_us - send_ts_us
        if self._last_transit_us is not None:
            d = abs(transit - self._last_transit_us)
            self._jitter += (d - self._jitter) * self.JITTER_GAIN
        self._last_transit_us = transit
        return transit * 1e-6

    @property
    def jitter_ms(self):
        return self._jitter / 1000.0

    def add_odom(self, sample):
        if sample.timestamp in self.odom_timestamps:
            return False
        self.odom_timestamps.add(sample.timestamp)
        if len(self.odom) == self.odom.maxlen and self.odom:
            self.odom_timestamps.discard(self.odom[0].timestamp)
        self.odom.append(sample)
        return True

    def odom_for(self, stamp, tolerance_s):
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

        # ── Benchmark report ────────────────────────────────────────────────
        self.declare_parameter("service_log_file", "/inf_server_service_log_udp_seq.csv")
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
        self.rcv_buffer_kb = gp("rcv_buffer_kb").get_parameter_value().integer_value
        self.session_timeout_s = gp("session_timeout_s").get_parameter_value().double_value
        self.stats_period_s = gp("stats_period_s").get_parameter_value().double_value
        self.status_log_period_s = gp("status_log_period_s").get_parameter_value().double_value
        self.odom_history = gp("odom_history").get_parameter_value().integer_value
        self.odom_tolerance_s = gp("odom_match_tolerance_s").get_parameter_value().double_value
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

        # Node-level mirrors of the per-session busy-drop counter, so a
        # windowed delta can be reported without depending on which sessions
        # are still alive at report time (a session can be reaped between rows).
        self._busy_drop_total = 0

        self._sessions = {}
        self._sessions_lock = threading.Lock()
        self._work_cv = threading.Condition(self._sessions_lock)

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
        self._busy_drop_at_report = 0
        self._bad_packets_at_report = 0
        self._open_sys_report()

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
        if self._report_writer is not None and self.sys_report_period_s > 0.0:
            self.create_timer(self.sys_report_period_s, self._write_sys_report_row)

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║ benchmark/udp: DR-SPAAM + KF Tracker     ║\n"
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
            f"{self.stats_period_s:g}s."
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
                f"is another server already bound to this port?")
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
            except Exception as e:
                self._bad_packets += 1
                self.get_logger().error(f"error handling datagram from {addr}: {e}",
                                        throttle_duration_sec=5.0)

    def _handle(self, data, addr):
        recv_us = wire.now_us()
        try:
            ver, ptype, _flags, sid, seq, send_us, body = wire.unpack_header(data)
        except wire.WireError:
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
        transit_s = session.note_arrival(send_us, recv_us)

        if ptype == wire.T_SCAN:
            # Scans only: odom and heartbeats ride their own cadence and would
            # dilute a figure meant to describe the scan path. Recorded before
            # _on_scan's accept/reject, because wire latency is a property of
            # what the link delivered, not of what we chose to process.
            self._perf.record_wire(transit_s)
            self._on_scan(session, seq, msg)
        elif ptype == wire.T_ODOM:
            self._on_odom(session, seq, msg)
        elif ptype == wire.T_HELLO:
            self._on_hello(session, msg)
        elif ptype == wire.T_HEARTBEAT:
            pass
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
            return
        view = ScanView(pkt, seq, time.monotonic())

        if session.scan_bins and len(view.ranges) != session.scan_bins:
            self.get_logger().warn(
                f"session {session.session_id:08x}: scan has {len(view.ranges)} bins, "
                f"HELLO announced {session.scan_bins}", throttle_duration_sec=10.0)

        with self._work_cv:
            if session.pending is not None:
                session.scans_dropped_busy += 1
                self._busy_drop_total += 1
            session.pending = view
            self._work_cv.notify()

    def _on_odom(self, session, seq, pkt):
        session.odom_seq.observe(seq)
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
        self._workers.append(threading.Thread(target=self._worker_loop, daemon=True))

    def _worker_loop(self):
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
        for session in self._sessions.values():
            if session.pending is not None:
                view, session.pending = session.pending, None
                return session, view
        return None

    def process_scan(self, session, view):
        dt = 0.1
        gap = None
        if session.last_scan_time is not None:
            # A gap here is real elapsed time -- dropped scans included -- so it
            # is passed to the filter rather than clamped to one scan period.
            # T_scan reports it raw, including values the filter refuses as dt:
            # a stalled or clock-jumped source is what it exists to show.
            gap = view.timestamp - session.last_scan_time
            if 0.0 < gap <= 2.0:
                dt = gap
        session.last_scan_time = view.timestamp

        odom = session.odom_for(view.timestamp, self.odom_tolerance_s)
        self.republish_scan(view)

        # Detection and tracking are timed separately even though they run
        # back to back here, so a sequential run reports the same T_det /
        # T_track split as the pipelined one and the two compare directly.
        t0 = time.perf_counter()
        with self._process_lock:
            dets_xy = self._detect(view)
            dets_xy, frame_id = self._to_tracking_frame(dets_xy, odom)
            t1 = time.perf_counter()
            active_tracks = session.tracker.step(dt, dets_xy)

        self._publish_detections_marker(dets_xy, frame_id)
        self._publish_ros(frame_id, active_tracks)
        t2 = time.perf_counter()

        # perf_counter() and monotonic() are the same underlying clock on
        # Linux CPython, which is what makes this cross-clock subtraction
        # valid; the pipelined variants already rely on the same fact.
        self._perf.record_detect(t1 - t0, scan_gap_s=gap)
        self._perf.record_track(t2 - t1, t2 - view.recv_mono)

        session.scans_processed += 1
        self._scan_count += 1
        self._scans_since_log += 1

    # ── stats / liveness ────────────────────────────────────────────────────

    def _send_all_stats(self):
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
        now = time.monotonic()
        with self._sessions_lock:
            dead = [sid for sid, s in self._sessions.items()
                    if now - s.last_seen > self.session_timeout_s]
            for sid in dead:
                self._sessions.pop(sid)
        for sid in dead:
            self.get_logger().info(
                f"session {sid:08x} timed out after {self.session_timeout_s:g}s of silence")

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        line = (f"[benchmark/udp] {len(sessions)} session(s) | "
                f"{rate:.1f} scans/s | {self.inference_fps:.1f} FPS (DR-SPAAM) | "
                f"{self._scan_count} total")
        if self._bad_packets or self._version_mismatches:
            line += (f" | {self._bad_packets} malformed, "
                     f"{self._version_mismatches} wrong-version")
        self.get_logger().info(
            line + "\n" + self._perf.console_block(self._perf.drain_console()))
        for s in sessions:
            self.get_logger().info(
                f"    {s.robot_name} [{s.session_id:08x}] "
                f"rx {s.scan_seq.received} lost {s.scan_seq.lost} "
                f"({s.scan_seq.loss_ratio * 100:.1f}%) reord {s.scan_seq.reordered} "
                f"busy-drop {s.scans_dropped_busy} | jitter {s.jitter_ms:.1f} ms "
                f"| odom {len(s.odom)}")

    # ── ROS output (shared with inf_server_udp) ─────────────────────────────

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
        elapsed_s = time.perf_counter() - t0
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

    def _to_tracking_frame(self, dets_xy, odom):
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

    # ── Resource / performance report ───────────────────────────────────────

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
        name = os.path.basename(self.service_log_file) or "inf_server_service_log_udp_seq.csv"
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
            "wall_time", "elapsed_s", "sessions",
            "scans_total", "scans_in_window", "scan_rate_hz", "drspaam_fps",
            # T_det / T_track / T_lat / T_scan / T_fwd + inference_ms_*,
            # the set every benchmark variant shares. T_lat here is socket
            # recv -> published. See sys_report.py for what each one covers.
            *TIMING_FIELDNAMES,
            # UDP-specific: loss/jitter averaged over sessions active at
            # report time, busy-drop and malformed-packet counts windowed
            "loss_percent", "jitter_ms",
            # Robot -> server transit, from the two wall-clock microsecond
            # stamps in the datagram header. An estimate; see sys_report.py.
            *WIRE_FIELDNAMES,
            "busy_drop_in_window", "bad_packets_in_window",
            "gpu_util_percent", "drspaam_gpu_duty_percent", "gpu_mem_util_percent",
            "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_proc_mem_mb",
            "torch_alloc_mb", "torch_reserved_mb", "torch_peak_mb",
            "gpu_power_w", "gpu_temp_c", "gpu_sm_clock_mhz",
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

        timings = self._perf.drain()
        inference_s = timings["inference"]
        scans = self._scan_count - self._scan_count_at_report
        self._scan_count_at_report = self._scan_count

        busy_window = self._busy_drop_total - self._busy_drop_at_report
        self._busy_drop_at_report = self._busy_drop_total
        bad_window = self._bad_packets - self._bad_packets_at_report
        self._bad_packets_at_report = self._bad_packets

        with self._sessions_lock:
            sessions = list(self._sessions.values())
        loss_pct = (round(100.0 * sum(s.scan_seq.loss_ratio for s in sessions) / len(sessions), 2)
                    if sessions else "")
        jitter_ms = (round(sum(s.jitter_ms for s in sessions) / len(sessions), 2)
                     if sessions else "")

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        row = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(now - self._report_t0, 3),
            "sessions": len(sessions),
            "scans_total": self._scan_count,
            "scans_in_window": scans,
            "scan_rate_hz": round(scans / window_s, 2),
            "drspaam_fps": round(len(inference_s) / sum(inference_s), 2) if inference_s else "",
            "loss_percent": loss_pct,
            "jitter_ms": jitter_ms,
            "busy_drop_in_window": busy_window,
            "bad_packets_in_window": bad_window,
            "drspaam_gpu_duty_percent": round(100.0 * sum(inference_s) / window_s, 2)
                                        if inference_s else "",
            "cpu_percent_system": psutil.cpu_percent(interval=None),
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
        row.update(PerfStats.timing_row(timings))
        row.update(PerfStats.wire_row(timings))
        row.update(self._gpu.sample())

        with self._report_lock:
            if self._report_writer is None:
                return
            self._report_writer.writerow(row)
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
            f"  UDP        T_lat {avg('t_lat_ms_mean'):.2f} ms | "
            f"wire~ {avg('wire_ms_mean'):.2f} ms | "
            f"loss {avg('loss_percent'):.1f}% | jitter {avg('jitter_ms'):.1f} ms\n"
            f"  GPU        util {avg('gpu_util_percent'):.1f}% | "
            f"mem {avg('gpu_mem_used_mb'):.0f} MB | {avg('gpu_power_w'):.1f} W\n"
            f"  Host       CPU {avg('cpu_percent_system'):.1f}% | "
            f"RAM {avg('ram_percent'):.1f}% | swap {avg('swap_percent'):.1f}%"
        )

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
        # After the socket is down, so no worker thread can still be recording.
        self._close_sys_report()
        self._gpu.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = UdpServerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        print(f"Error starting benchmark udp node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
