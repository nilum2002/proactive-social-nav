"""End-to-end tests: the real node, a real socket, a mock inf_server_udp.

These run the actual ROS node -- real subscriptions, real datagrams over
loopback -- and assert on what a server would receive. dr_spaam is not needed,
so unlike the server half this side is fully testable on the robot itself.
"""
import socket
import time

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from inf_client_udp import wire
from inf_client_udp.udp_client_node import InfClientUdpNode

BINS = 450


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def server():
    """A mock inf_server_udp: just a bound socket that can also reply."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    yield sock
    sock.close()


@pytest.fixture
def client(server):
    node = InfClientUdpNode(parameter_overrides=[
        Parameter("server_address", value=f"127.0.0.1:{server.getsockname()[1]}"),
        Parameter("robot_name", value="test-bot"),
        Parameter("status_log_period_s", value=0.0),
        Parameter("odom_max_rate_hz", value=0.0),   # no throttle, so tests are deterministic
        Parameter("odom_redundancy", value=3),
    ])
    yield node
    node.shutdown()
    node.destroy_node()


def _pump(*nodes, cycles=25):
    for _ in range(cycles):
        for n in nodes:
            rclpy.spin_once(n, timeout_sec=0.01)


def _scan_msg(fill=3.5, bins=BINS):
    msg = LaserScan()
    msg.header.stamp.sec = 100
    msg.header.stamp.nanosec = 500_000_000
    msg.angle_min = -np.pi
    msg.angle_max = float(np.pi)
    msg.angle_increment = float(2 * np.pi / bins)
    msg.range_min = 0.05
    msg.range_max = 8.0
    msg.ranges = [float(fill)] * bins
    return msg


def _odom_msg(ts, x):
    msg = Odometry()
    msg.header.stamp.sec = int(ts)
    msg.header.stamp.nanosec = int((ts % 1) * 1e9)
    msg.pose.pose.position.x = x
    msg.pose.pose.orientation.w = 1.0
    return msg


def _recv(server, want_type, tries=6):
    """Read datagrams until one of `want_type` shows up (skipping HELLO etc.)."""
    for _ in range(tries):
        data = server.recv(wire.RECV_BUFFER_BYTES)
        hdr = wire.unpack_header(data)
        if hdr[1] == want_type:
            return data, hdr, wire.decode_payload(hdr[1], hdr[6])
    raise AssertionError(f"no {wire.TYPE_NAMES[want_type]} packet arrived")


# ── scans ───────────────────────────────────────────────────────────────────
def test_scan_reaches_the_server_intact_and_in_one_datagram(client, server):
    pub = rclpy.create_node("fake_lidar")
    p = pub.create_publisher(LaserScan, "/scan", 10)
    _pump(client, pub, cycles=10)          # let discovery settle
    p.publish(_scan_msg(fill=3.5))
    _pump(client, pub)

    data, hdr, scan = _recv(server, wire.T_SCAN)
    ver, ptype, _flags, sid, seq, _us, _body = hdr

    assert len(data) <= wire.SAFE_DATAGRAM_BYTES, "a scan must never fragment"
    assert ver == wire.PROTO_VERSION
    assert sid == client._session_id
    assert seq == 1
    assert scan.timestamp == pytest.approx(100.5)
    assert len(scan.ranges_mm) == 2 * BINS
    assert np.allclose(wire.dequantize_ranges(scan.ranges_mm), 3.5, atol=1e-3)
    pub.destroy_node()


def test_no_return_beams_survive_the_round_trip(client, server):
    pub = rclpy.create_node("fake_lidar2")
    p = pub.create_publisher(LaserScan, "/scan", 10)
    _pump(client, pub, cycles=10)
    msg = _scan_msg()
    msg.ranges = [float("inf")] * 10 + [2.0] * (BINS - 10)
    p.publish(msg)
    _pump(client, pub)

    _data, _hdr, scan = _recv(server, wire.T_SCAN)
    out = wire.dequantize_ranges(scan.ranges_mm)
    assert np.all(np.isinf(out[:10]))
    assert np.allclose(out[10:], 2.0, atol=1e-3)
    pub.destroy_node()


def test_scan_sequence_increments_so_the_server_can_see_gaps(client, server):
    pub = rclpy.create_node("fake_lidar3")
    p = pub.create_publisher(LaserScan, "/scan", 10)
    _pump(client, pub, cycles=10)
    for _ in range(3):
        p.publish(_scan_msg())
        _pump(client, pub, cycles=8)

    seqs = [_recv(server, wire.T_SCAN)[1][4] for _ in range(3)]
    assert seqs == [1, 2, 3]
    pub.destroy_node()


# ── odometry redundancy ─────────────────────────────────────────────────────
def test_odom_packets_carry_redundant_history_newest_first(client, server):
    """Losing a pose should take several consecutive drops, not one."""
    pub = rclpy.create_node("fake_odom")
    p = pub.create_publisher(Odometry, "/odom", 10)
    _pump(client, pub, cycles=10)
    for i in range(3):
        p.publish(_odom_msg(200.0 + i, x=float(i)))
        _pump(client, pub, cycles=8)

    packets = [_recv(server, wire.T_ODOM)[2] for _ in range(3)]
    assert [len(pk.samples) for pk in packets] == [1, 2, 3]
    newest = packets[-1].samples
    assert [s.x for s in newest] == [2.0, 1.0, 0.0], "samples[0] must be the newest"
    assert newest[0].timestamp > newest[-1].timestamp
    pub.destroy_node()


# ── session ─────────────────────────────────────────────────────────────────
def test_hello_goes_out_before_any_sensor_data(client, server):
    """Regression: HELLO used to wait for the first keepalive tick, so a server
    that replied to our very first scan flipped _server_seen before HELLO had
    ever been sent and the robot stayed anonymous for the whole session."""
    data = server.recv(wire.RECV_BUFFER_BYTES)
    hdr = wire.unpack_header(data)
    assert hdr[1] == wire.T_HELLO, "the very first datagram must be HELLO"
    assert wire.decode_payload(hdr[1], hdr[6]).robot_name == "test-bot"


def test_scan_geometry_is_announced_once_it_is_known(client, server):
    """The startup HELLO cannot carry bins/frame/rate -- no scan has arrived
    yet -- so a second HELLO must follow once they are measurable."""
    pub = rclpy.create_node("fake_lidar5")
    p = pub.create_publisher(LaserScan, "/scan", 10)
    _pump(client, pub, cycles=10)
    for i in range(3):
        msg = _scan_msg()
        msg.header.frame_id = "laser"
        msg.header.stamp.sec = 100 + i          # 1 Hz, so a rate is measurable
        p.publish(msg)
        _pump(client, pub, cycles=8)

    hellos = []
    for _ in range(12):
        try:
            hdr = wire.unpack_header(server.recv(wire.RECV_BUFFER_BYTES))
        except socket.timeout:
            break
        if hdr[1] == wire.T_HELLO:
            hellos.append(wire.decode_payload(hdr[1], hdr[6]))
    assert len(hellos) >= 2, "geometry must be re-announced after the first scans"
    assert hellos[-1].scan_bins == BINS
    assert hellos[-1].laser_frame_id == "laser"
    assert hellos[-1].scan_rate_hz == pytest.approx(1.0, abs=0.01)
    pub.destroy_node()


def test_hello_is_repeated_until_the_server_answers(client, server):
    """HELLO is a datagram like any other and can be lost, so it must not be
    sent once and assumed delivered."""
    client._keepalive()
    client._keepalive()
    for _ in range(2):
        _data, hdr, hello = _recv(server, wire.T_HELLO, tries=3)
        assert hello.robot_name == "test-bot"
        assert hdr[3] == client._session_id


def test_stats_reply_marks_the_server_seen_and_stops_hello(client, server):
    client._keepalive()
    _recv(server, wire.T_HELLO, tries=3)

    stats = wire.pb.StatsPacket(scans_received=100, scans_lost=0, loss_ratio=0.0,
                                inference_fps=20.0, active_tracks=2)
    server.sendto(wire.pack(wire.T_STATS, client._session_id, 1, stats),
                  client._sock.getsockname())

    deadline = time.monotonic() + 3.0
    while not client._server_seen and time.monotonic() < deadline:
        time.sleep(0.02)
    assert client._server_seen, "client should notice the server's STATS reply"


def test_bye_is_sent_on_shutdown(client, server):
    client.shutdown()
    _data, _hdr, bye = _recv(server, wire.T_BYE, tries=4)
    assert "shut" in bye.reason.lower()


# ── rate adaptation ─────────────────────────────────────────────────────────
def test_high_reported_loss_backs_the_scan_rate_off(client):
    """UDP has no congestion control; this is the only brake in the system."""
    assert client._decimation == 1
    client._adapt(wire.pb.StatsPacket(scans_received=100, scans_lost=30, loss_ratio=0.30))
    assert client._decimation == 2
    client._adapt(wire.pb.StatsPacket(scans_received=100, scans_lost=30, loss_ratio=0.30))
    assert client._decimation == 3


def test_recovery_is_gradual_and_bounded(client):
    for _ in range(10):
        client._adapt(wire.pb.StatsPacket(scans_received=100, scans_lost=90, loss_ratio=0.90))
    assert client._decimation == client.max_decimation, "must not back off without limit"

    for _ in range(10):
        client._adapt(wire.pb.StatsPacket(scans_received=100, scans_lost=0, loss_ratio=0.0))
    assert client._decimation == 1, "must recover all the way once the link is clean"


def test_a_busy_server_also_triggers_back_off(client):
    """scans_dropped_busy means the GPU is behind, not that the link is bad --
    but sending faster helps neither, so it backs off the same way."""
    client._adapt(wire.pb.StatsPacket(scans_received=100, scans_lost=0, loss_ratio=0.0,
                                      scans_dropped_busy=40))
    assert client._decimation == 2


def test_decimation_actually_skips_scans(client, server):
    pub = rclpy.create_node("fake_lidar4")
    p = pub.create_publisher(LaserScan, "/scan", 10)
    _pump(client, pub, cycles=10)
    client._decimation = 3

    for _ in range(9):
        p.publish(_scan_msg())
        _pump(client, pub, cycles=6)

    assert client._scans_in == 9
    assert client._scans_sent == 3
    assert client._scans_decimated == 6
    pub.destroy_node()
