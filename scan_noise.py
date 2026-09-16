#!/usr/bin/env python3
"""Quantify scan-to-scan instability on a LaserScan topic.

Run it with the robot and the scene completely still, on both ends, and
compare the two reports:

    robot  : python3 scan_noise.py /scan
    server : python3 scan_noise.py /inf_server_node/scan

Same numbers on both sides  -> the sensor/driver is producing it; nothing in
                               the transport or the server is at fault.
Clean on the robot, noisy on the server -> the relay path introduced it, and
                               the transport in use is where to look.

Two independent causes of what reads as "vibration" in RViz are measured
separately, because they need completely different fixes:

  * RANGE noise -- the same beam reports a different distance each scan. Shows
    up as points twitching radially (in/out from the sensor).

  * GEOMETRY jitter -- angle_min / angle_increment / bin count change between
    scans, so every point is drawn at a slightly different bearing even when
    the measured distances are rock solid. Shows up as the whole scan
    shimmering rotationally, and is common on cheap spinning LiDARs whose
    motor speed varies: the driver reports the true start angle and the true
    points-per-revolution of each turn, and both wander.
"""
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanNoise(Node):

    def __init__(self, topic, n_frames):
        super().__init__("scan_noise")
        self.n_frames = n_frames
        self.ranges = []
        self.angle_min = []
        self.angle_inc = []
        self.stamps = []

        # BEST_EFFORT matches both a BEST_EFFORT driver and a RELIABLE
        # republisher; a RELIABLE subscriber would silently match nothing at
        # all against the former, which looks identical to "topic is dead".
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(LaserScan, topic, self._cb, qos)
        self.get_logger().info(
            f"collecting {n_frames} frames from {topic} -- keep everything still")

    def _cb(self, msg):
        if self.done:
            return
        self.ranges.append(np.asarray(msg.ranges, dtype=np.float64))
        self.angle_min.append(msg.angle_min)
        self.angle_inc.append(msg.angle_increment)
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        n = len(self.ranges)
        if n % 25 == 0:
            self.get_logger().info(f"  {n}/{self.n_frames}")

    @property
    def done(self):
        return len(self.ranges) >= self.n_frames


def report(node, topic):
    bins = np.array([len(r) for r in node.ranges])
    modal = int(np.bincount(bins).argmax())

    print(f"\n=== {topic} : {len(node.ranges)} frames ===\n")

    # ── geometry ────────────────────────────────────────────────────────────
    amin = np.array(node.angle_min)
    ainc = np.array(node.angle_inc)
    amin_spread_mrad = (amin.max() - amin.min()) * 1e3
    ainc_spread_urad = (ainc.max() - ainc.min()) * 1e6

    print("GEOMETRY (a moving scan with perfectly stable ranges lives here)")
    print(f"  bin count      : {modal}"
          + ("" if bins.min() == bins.max()
             else f"   VARIES {bins.min()}..{bins.max()}  <-- rotational shimmer"))
    print(f"  angle_min      : spread {amin_spread_mrad:8.3f} mrad"
          f"   (= {amin_spread_mrad * 3.0:6.2f} mm of sideways sweep at 3 m)")
    print(f"  angle_increment: spread {ainc_spread_urad:8.3f} urad"
          f"   (accumulates across the sweep)")

    # ── timing ──────────────────────────────────────────────────────────────
    if len(node.stamps) > 2:
        dt_ms = np.diff(np.array(node.stamps)) * 1e3
        print(f"  scan period    : {dt_ms.mean():6.2f} ms mean, "
              f"{dt_ms.std():5.2f} ms std, {dt_ms.max():6.2f} ms max")

    # ── range noise ─────────────────────────────────────────────────────────
    a = np.vstack([r for r in node.ranges if len(r) == modal])
    a[~np.isfinite(a)] = np.nan
    n_valid = np.isfinite(a).sum(axis=0)
    # A beam that only sometimes returns says more about the surface than
    # about stability, so only beams answering most of the time are scored.
    keep = n_valid >= 0.8 * a.shape[0]

    print(f"\nRANGE NOISE  ({keep.sum()} of {modal} beams return >=80% of the time)")
    if not keep.any():
        print("  no consistently-returning beams -- aim at a wall and retry")
        return

    with np.errstate(invalid="ignore"):
        std_mm = np.nanstd(a[:, keep], axis=0) * 1e3
        mean_m = np.nanmean(a[:, keep], axis=0)
    pct = 100.0 * (std_mm * 1e-3) / np.maximum(mean_m, 1e-6)

    print(f"  per-beam std   : median {np.median(std_mm):6.2f} mm, "
          f"p95 {np.percentile(std_mm, 95):6.2f} mm, max {std_mm.max():7.2f} mm")
    print(f"  as % of range  : median {np.median(pct):5.2f} %, "
          f"p95 {np.percentile(pct, 95):5.2f} %")
    print(f"  beams >10mm std: {(std_mm > 10).sum()} of {keep.sum()} "
          f"({100.0 * (std_mm > 10).mean():.1f} %)")

    print("\n  Reference: this repo's wire.py notes the LD19 is +/-1-2% accurate,")
    print("  so a median around 1-2% of range is the sensor behaving normally.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    topic = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    rclpy.init()
    node = ScanNoise(topic, n_frames)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=1.0)
        report(node, topic)
    except KeyboardInterrupt:
        if node.ranges:
            report(node, topic)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
