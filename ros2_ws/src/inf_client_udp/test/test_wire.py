"""Unit tests for the UDP wire format.

wire.py and perception_udp.proto are byte-identical copies of the files in
inf_server_udp: they are the one thing in this system that must agree exactly
with a program running on a different machine. That makes them worth pinning
down with tests rather than with a successful run.
"""
import math

import numpy as np
import pytest

from inf_client_udp import wire


# ── header ──────────────────────────────────────────────────────────────────
def test_header_roundtrip():
    pkt = wire.pack(wire.T_HEARTBEAT, 0xDEADBEEF, 42,
                    wire.pb.HeartbeatPacket(timestamp=123.5), send_ts_us=99)
    ver, ptype, flags, sid, seq, send_us, body = wire.unpack_header(pkt)
    assert (ver, ptype, sid, seq, send_us) == (wire.PROTO_VERSION, wire.T_HEARTBEAT,
                                               0xDEADBEEF, 42, 99)
    assert wire.decode_payload(ptype, body).timestamp == 123.5


def test_runt_and_garbage_are_rejected_not_crashed():
    with pytest.raises(wire.WireError):
        wire.unpack_header(b"\x00" * 4)
    with pytest.raises(wire.WireError):
        wire.unpack_header(b"XX" + b"\x00" * 30)


def test_unknown_type_is_rejected():
    with pytest.raises(wire.WireError):
        wire.decode_payload(99, b"")


# ── the MTU guarantee ───────────────────────────────────────────────────────
def test_450_bin_scan_fits_one_datagram():
    """The whole design rests on this: one scan is one datagram, always.

    A scan split across two IP fragments is lost if either fragment is lost,
    which roughly doubles the effective loss rate for free.
    """
    ranges = np.random.uniform(0.1, 8.0, 450)
    pkt = wire.pack(wire.T_SCAN, 1, 1, wire.pb.ScanPacket(
        timestamp=1.0, angle_min=-math.pi, angle_max=math.pi,
        angle_increment=2 * math.pi / 450, range_min=0.05, range_max=8.0,
        ranges_mm=wire.quantize_ranges(ranges)))
    assert len(pkt) <= wire.SAFE_DATAGRAM_BYTES
    assert len(pkt) <= wire.scan_datagram_bytes(450)  # the bound must not undershoot


def test_scan_size_is_independent_of_scene_content():
    """The reason ranges_mm is `bytes` and not `repeated uint32`.

    An all-far-away scan and an all-no-return scan must produce identical packet
    sizes, otherwise datagram size would depend on what the room looks like and
    the MTU guarantee would hold only until the robot drove into a corridor.
    """
    def size(fill):
        return len(wire.pack(wire.T_SCAN, 1, 1, wire.pb.ScanPacket(
            timestamp=1.0, range_max=8.0,
            ranges_mm=wire.quantize_ranges(np.full(450, fill)))))
    assert size(7.999) == size(0.05) == size(float("inf"))


def test_odom_with_redundancy_is_still_tiny():
    samples = [wire.pb.OdomSample(timestamp=100.0 + i, x=1.5, y=-2.5, qz=0.1, qw=0.99,
                                  vx=0.4, vtheta=0.1) for i in range(3)]
    pkt = wire.pack(wire.T_ODOM, 1, 1, wire.pb.OdomPacket(samples=samples))
    assert len(pkt) < 250   # three poses cost less than a fifth of one scan


# ── quantization ────────────────────────────────────────────────────────────
def test_quantization_is_mm_accurate():
    ranges = np.array([0.05, 1.234, 3.5, 7.999], dtype=np.float32)
    out = wire.dequantize_ranges(wire.quantize_ranges(ranges))
    assert np.allclose(out, ranges, atol=1e-3)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1.0, 0.0, 1e6])
def test_no_return_values_all_map_to_inf(bad):
    """inf, not NaN and not 0, so the array behaves like a real LaserScan and
    the server's existing inf-handling applies unchanged."""
    out = wire.dequantize_ranges(wire.quantize_ranges([bad]))
    assert np.isinf(out[0])


def test_odd_length_blob_is_rejected():
    with pytest.raises(wire.WireError):
        wire.dequantize_ranges(b"\x01\x02\x03")
