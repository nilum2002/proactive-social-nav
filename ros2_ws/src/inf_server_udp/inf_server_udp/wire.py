"""Wire format shared by inf_client_udp (robot) and inf_server_udp (off-board).

This module is the contract. It must stay byte-identical on both ends -- copy it
alongside perception_udp.proto, never fork it.

Layout of every datagram:

    +----------------------------- 21 bytes -----------------------------+
    | magic(2) | ver(1) | type(1) | flags(1) | session(4) | seq(4) | us(8) |
    +---------------------------------------------------------------------+
    | protobuf message, class selected by `type`                          |
    +---------------------------------------------------------------------+

The header is raw struct rather than protobuf on purpose. A receiver must be
able to read `ver` and `type` and decide to drop a packet *without* attempting a
decode: a version-mismatched protobuf frequently parses "successfully" into
garbage instead of raising, so checking a plain version byte first is what makes
a mismatch loud rather than subtly wrong.

Sequence numbers live in a separate space per packet type, so a gap in the scan
stream is not masked by odom traffic arriving at a different rate.

The generated module is imported relatively (`from . import
perception_udp_pb2`) so this file is genuinely byte-identical in both packages.
inf_client/inf_server need a sys.path.insert dance only because protoc's *gRPC*
stub emits an absolute `import perception_stream_pb2`; there is no service here,
so the message module is self-contained and a relative import works.
"""
import struct
import time

import numpy as np

from . import perception_udp_pb2 as pb

# ── Header ──────────────────────────────────────────────────────────────────
MAGIC = b"PS"
PROTO_VERSION = 1

# little-endian, no padding: magic, version, type, flags, session, seq, send_ts_us
_HEADER = struct.Struct("<2sBBBIIQ")
HEADER_SIZE = _HEADER.size  # 21

# ── Packet types ────────────────────────────────────────────────────────────
T_HELLO = 1
T_SCAN = 2
T_ODOM = 3
T_HEARTBEAT = 4
T_STATS = 5
T_BYE = 6

TYPE_NAMES = {
    T_HELLO: "HELLO", T_SCAN: "SCAN", T_ODOM: "ODOM",
    T_HEARTBEAT: "HEARTBEAT", T_STATS: "STATS", T_BYE: "BYE",
}

PAYLOAD_TYPES = {
    T_HELLO: pb.HelloPacket,
    T_SCAN: pb.ScanPacket,
    T_ODOM: pb.OdomPacket,
    T_HEARTBEAT: pb.HeartbeatPacket,
    T_STATS: pb.StatsPacket,
    T_BYE: pb.ByePacket,
}

# ── Sizing ──────────────────────────────────────────────────────────────────
# 1500 MTU - 20 IP - 8 UDP = 1472. We stay under 1400 so the same packets still
# fit once someone runs this inside WireGuard (~60 B) or a VPN, without having
# to revisit the format. A 450-bin scan is ~958 B, leaving real headroom.
SAFE_DATAGRAM_BYTES = 1400
# Comfortably over SAFE_DATAGRAM_BYTES so an oversized packet is seen as
# oversized (and logged) rather than silently truncated by recvfrom.
RECV_BUFFER_BYTES = 2048

NO_RETURN = 0          # sentinel in ranges_mm; 0 mm is below any real range_min
MAX_RANGE_MM = 65535   # 65.535 m -- far past any 2D indoor LiDAR


class WireError(ValueError):
    """Malformed datagram: bad magic, short header, or unknown type."""


def now_us():
    """Wall-clock microseconds, matched to ROS message stamps (not monotonic)."""
    return int(time.time() * 1e6)


def pack(ptype, session_id, seq, payload, flags=0, send_ts_us=None):
    """Serialize one datagram. Returns bytes ready for sendto()."""
    if send_ts_us is None:
        send_ts_us = now_us()
    header = _HEADER.pack(
        MAGIC, PROTO_VERSION, ptype, flags,
        session_id & 0xFFFFFFFF, seq & 0xFFFFFFFF,
        send_ts_us & 0xFFFFFFFFFFFFFFFF,
    )
    return header + payload.SerializeToString()


def unpack_header(datagram):
    """Validate and split off the header.

    Returns (ver, ptype, flags, session_id, seq, send_ts_us, payload_bytes).
    Raises WireError for anything that is not one of our packets -- an open UDP
    port receives whatever the network sends it, including port scans and
    strays, so this is a normal condition and not a reason to crash.
    """
    if len(datagram) < HEADER_SIZE:
        raise WireError(f"runt datagram: {len(datagram)} B < {HEADER_SIZE} B header")
    magic, ver, ptype, flags, session_id, seq, send_ts_us = _HEADER.unpack_from(datagram)
    if magic != MAGIC:
        raise WireError(f"bad magic {magic!r}")
    return ver, ptype, flags, session_id, seq, send_ts_us, datagram[HEADER_SIZE:]


def decode_payload(ptype, payload_bytes):
    """Parse the protobuf body for an already-validated header type."""
    cls = PAYLOAD_TYPES.get(ptype)
    if cls is None:
        raise WireError(f"unknown packet type {ptype}")
    msg = cls()
    msg.ParseFromString(payload_bytes)
    return msg


# ── Range quantization ──────────────────────────────────────────────────────
def quantize_ranges(ranges):
    """float32 metres -> little-endian uint16 millimetres, 0 for no-return.

    NaN, inf, negatives and anything past 65.535 m all collapse to the no-return
    sentinel: the client's job is to say "this beam gave nothing", and it stays
    the server's job to decide what that means for detection.
    """
    a = np.asarray(ranges, dtype=np.float64)
    finite = np.isfinite(a)
    # Multiply only where finite; inf/NaN would otherwise poison the rint and
    # make the uint16 cast undefined.
    mm = np.zeros(a.shape, dtype=np.float64)
    np.multiply(a, 1000.0, out=mm, where=finite)
    np.rint(mm, out=mm)
    valid = finite & (mm >= 1.0) & (mm <= MAX_RANGE_MM)
    return np.where(valid, mm, NO_RETURN).astype("<u2").tobytes()


def dequantize_ranges(blob):
    """little-endian uint16 millimetres -> float32 metres, inf for no-return.

    Returns inf (not NaN, not 0) for missing beams so the array behaves exactly
    like a real LaserScan.ranges from the driver and the detector's existing
    inf-handling applies unchanged.
    """
    if len(blob) % 2:
        raise WireError(f"ranges_mm has odd length {len(blob)}")
    q = np.frombuffer(blob, dtype="<u2")
    out = q.astype(np.float32) / 1000.0
    out[q == NO_RETURN] = np.inf
    return out


def scan_datagram_bytes(n_bins):
    """Upper bound on the datagram size for an n-bin scan, for an MTU check.

    21 header + protobuf body: 1 double (9) + 5 floats (5*5) + the bytes field
    (1 tag + 2 length varint + 2*n). Real packets come in a few bytes under this
    because proto3 omits zero-valued scalars -- erring high is the safe
    direction for a bound whose whole job is to keep us off the MTU.
    """
    return HEADER_SIZE + 9 + 25 + 3 + 2 * n_bins
