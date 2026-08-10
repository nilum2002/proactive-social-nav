import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker

from dr_spaam.detector import Detector


class DrSpaamROS2(Node):
    """ROS2 node to detect pedestrians using DROW3 or DR-SPAAM.

    Enhanced with diagnostic logging to expose raw confidence scores,
    scan health statistics, and per-frame detection breakdowns.
    """

    def __init__(self):
        super().__init__("dr_spaam_ros2_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("weight_file", "")
        self.declare_parameter("detector_model", "DR-SPAAM")
        self.declare_parameter("conf_thresh", 0.8)
        self.declare_parameter("stride", 1)
        self.declare_parameter("panoramic_scan", True)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("gpu", True)
        # Diagnostics: log every N scans (0 = every scan, high value = less spam)
        self.declare_parameter("log_every_n_scans", 10)
        # Diagnostics: also log raw scores above this floor (before conf_thresh)
        self.declare_parameter("raw_score_floor", 0.1)

        self.weight_file    = self.get_parameter("weight_file").get_parameter_value().string_value
        self.detector_model = self.get_parameter("detector_model").get_parameter_value().string_value
        self.conf_thresh    = self.get_parameter("conf_thresh").get_parameter_value().double_value
        self.stride         = self.get_parameter("stride").get_parameter_value().integer_value
        self.panoramic_scan = self.get_parameter("panoramic_scan").get_parameter_value().bool_value
        self.scan_topic     = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.use_gpu        = self.get_parameter("gpu").get_parameter_value().bool_value
        self.log_every_n    = self.get_parameter("log_every_n_scans").get_parameter_value().integer_value
        self.raw_score_floor = self.get_parameter("raw_score_floor").get_parameter_value().double_value

        # ── Validate ─────────────────────────────────────────────────────────
        if not self.weight_file:
            self.get_logger().error(
                "Parameter 'weight_file' is empty! Please provide a valid checkpoint path."
            )
            raise ValueError("weight_file parameter is empty.")

        # ── Init detector ─────────────────────────────────────────────────────
        self.get_logger().info(
            f"Initializing '{self.detector_model}' from: {self.weight_file}"
        )
        self._detector = Detector(
            self.weight_file,
            model=self.detector_model,
            gpu=self.use_gpu,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._dets_pub  = self.create_publisher(PoseArray, "~/detections", 10)
        self._rviz_pub  = self.create_publisher(Marker, "~/rviz_marker", 10)

        # ── Subscriber ────────────────────────────────────────────────────────
        self._scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            10
        )

        # ── Diagnostics state ─────────────────────────────────────────────────
        self._scan_count       = 0   # total scans processed
        self._total_raw_dets   = 0   # raw detections before threshold
        self._total_conf_dets  = 0   # detections after threshold
        self._fov_deg          = None

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║        DR-SPAAM Diagnostics Active       ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  conf_thresh    : {self.conf_thresh}\n"
            f"  raw_score_floor: {self.raw_score_floor}  (scores above this are logged)\n"
            f"  log_every_n    : {self.log_every_n}  scans\n"
            f"  Subscribed to  : {self.scan_topic}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _scan_callback(self, msg: LaserScan):
        self._scan_count += 1

        # ── 1. Configure FOV on first scan ───────────────────────────────────
        if not self._detector.is_ready():
            fov_rad = msg.angle_increment * len(msg.ranges)
            self._fov_deg = np.rad2deg(fov_rad)
            self._detector.set_laser_fov(self._fov_deg)
            self.get_logger().info(
                f"[DIAG] FOV configured: {self._fov_deg:.2f}°  |  "
                f"Points: {len(msg.ranges)}  |  "
                f"angle_increment: {np.rad2deg(msg.angle_increment):.3f}°  |  "
                f"Range: [{msg.range_min:.2f}, {msg.range_max:.2f}] m"
            )

        # ── 2. Build scan array ───────────────────────────────────────────────
        raw_ranges = np.array(msg.ranges, dtype=np.float32)

        # Scan health stats (BEFORE clipping)
        valid_mask   = np.isfinite(raw_ranges) & (raw_ranges >= msg.range_min) & (raw_ranges <= msg.range_max)
        n_valid      = int(valid_mask.sum())
        n_inf        = int(np.isinf(raw_ranges).sum())
        n_nan        = int(np.isnan(raw_ranges).sum())
        n_out_range  = int(len(raw_ranges) - n_valid - n_inf - n_nan)
        pct_valid    = 100.0 * n_valid / len(raw_ranges) if len(raw_ranges) > 0 else 0.0

        if n_valid > 0:
            valid_ranges = raw_ranges[valid_mask]
            min_r, max_r, mean_r = valid_ranges.min(), valid_ranges.max(), valid_ranges.mean()
        else:
            min_r = max_r = mean_r = float("nan")

        # Clip for inference
        scan = raw_ranges.copy()
        scan[scan < msg.range_min] = 29.99
        scan[scan > msg.range_max] = 29.99
        scan[np.isinf(scan)]       = 29.99
        scan[np.isnan(scan)]       = 29.99

        scan_phi = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment

        # ── 3. Inference ──────────────────────────────────────────────────────
        dets_xy, dets_cls, _ = self._detector(scan, scan_phi=scan_phi)
        dets_cls_flat = dets_cls.reshape(-1)

        # Raw detections above the logging floor (BEFORE conf_thresh)
        raw_above_floor = dets_cls_flat[dets_cls_flat >= self.raw_score_floor]
        n_raw           = len(raw_above_floor)

        # ── 4. Apply confidence threshold ─────────────────────────────────────
        conf_mask  = (dets_cls_flat >= self.conf_thresh)
        dets_xy_f  = dets_xy[conf_mask]
        dets_cls_f = dets_cls_flat[conf_mask]
        n_conf     = len(dets_cls_f)

        self._total_raw_dets  += n_raw
        self._total_conf_dets += n_conf

        # ── 5. Publish ────────────────────────────────────────────────────────
        dets_msg = self._detections_to_pose_array(dets_xy_f, dets_cls_f)
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

        rviz_msg = self._detections_to_rviz_marker(dets_xy_f, dets_cls_f)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)

        # ── 6. Diagnostic logging ─────────────────────────────────────────────
        if self.log_every_n <= 0 or (self._scan_count % self.log_every_n == 0):
            self._log_diagnostics(
                msg, n_valid, n_inf, n_nan, n_out_range, pct_valid,
                min_r, max_r, mean_r,
                dets_cls_flat, raw_above_floor, n_raw,
                dets_xy_f, dets_cls_f, n_conf,
            )

    # ─────────────────────────────────────────────────────────────────────────
    def _log_diagnostics(
        self, msg,
        n_valid, n_inf, n_nan, n_out_range, pct_valid,
        min_r, max_r, mean_r,
        all_scores, raw_above_floor, n_raw,
        dets_xy_f, dets_cls_f, n_conf,
    ):
        sep = "─" * 56

        # ── Scan health ───────────────────────────────────────────────────────
        health_status = "✅ GOOD" if pct_valid > 50 else ("⚠️  SPARSE" if pct_valid > 10 else "❌ VERY SPARSE")
        scan_block = (
            f"\n  ┌─ SCAN HEALTH  [frame #{self._scan_count}]  {sep[:14]}\n"
            f"  │  frame_id    : {msg.header.frame_id}\n"
            f"  │  total pts   : {len(msg.ranges)}\n"
            f"  │  valid pts   : {n_valid}  ({pct_valid:.1f}%)  {health_status}\n"
            f"  │  inf/nan/oob : {n_inf} / {n_nan} / {n_out_range}\n"
            f"  │  range stats : min={min_r:.3f}m  max={max_r:.3f}m  mean={mean_r:.3f}m\n"
            f"  └{'─'*58}"
        )

        # ── Raw confidence scores (before threshold) ──────────────────────────
        if n_raw > 0:
            sorted_scores = np.sort(raw_above_floor)[::-1]
            top_scores_str = "  ".join(f"{s:.3f}" for s in sorted_scores[:10])
            scores_block = (
                f"\n  ┌─ RAW SCORES (≥{self.raw_score_floor:.2f}, before conf_thresh={self.conf_thresh})  ──\n"
                f"  │  candidates above floor : {n_raw}\n"
                f"  │  top-10 scores          : {top_scores_str}\n"
                f"  │  score distribution:\n"
                f"  │    [0.10–0.30): {int(((raw_above_floor>=0.10)&(raw_above_floor<0.30)).sum())}\n"
                f"  │    [0.30–0.50): {int(((raw_above_floor>=0.30)&(raw_above_floor<0.50)).sum())}\n"
                f"  │    [0.50–0.70): {int(((raw_above_floor>=0.50)&(raw_above_floor<0.70)).sum())}\n"
                f"  │    [0.70–0.90): {int(((raw_above_floor>=0.70)&(raw_above_floor<0.90)).sum())}\n"
                f"  │    [0.90–1.00]: {int(((raw_above_floor>=0.90)&(raw_above_floor<=1.00)).sum())}\n"
                f"  └{'─'*58}"
            )
        else:
            scores_block = (
                f"\n  ┌─ RAW SCORES ─────────────────────────────────────────\n"
                f"  │  ⚠️  NO scores above floor ({self.raw_score_floor:.2f})\n"
                f"  │  This means the model sees NO person-like clusters.\n"
                f"  │  → Check if person is within LiDAR range & scan plane.\n"
                f"  └{'─'*58}"
            )

        # ── Accepted detections (after threshold) ─────────────────────────────
        if n_conf > 0:
            det_lines = "\n".join(
                f"  │    [{i+1}] score={dets_cls_f[i]:.4f}  "
                f"x={dets_xy_f[i,0]:+.3f}m  y={dets_xy_f[i,1]:+.3f}m  "
                f"dist={np.hypot(dets_xy_f[i,0], dets_xy_f[i,1]):.3f}m"
                for i in range(n_conf)
            )
            det_block = (
                f"\n  ┌─ DETECTIONS (conf≥{self.conf_thresh}) ────────────────────────\n"
                f"  │  count : {n_conf}\n"
                f"{det_lines}\n"
                f"  └{'─'*58}"
            )
        else:
            det_block = (
                f"\n  ┌─ DETECTIONS (conf≥{self.conf_thresh}) ────────────────────────\n"
                f"  │  count : 0  — no detections above threshold\n"
                f"  └{'─'*58}"
            )

        # ── Running summary ───────────────────────────────────────────────────
        avg_raw  = self._total_raw_dets  / self._scan_count
        avg_conf = self._total_conf_dets / self._scan_count
        summary_block = (
            f"\n  ┌─ RUNNING SUMMARY ────────────────────────────────────\n"
            f"  │  scans processed : {self._scan_count}\n"
            f"  │  avg raw cands/scan  : {avg_raw:.2f}\n"
            f"  │  avg confirmed/scan  : {avg_conf:.2f}\n"
            f"  └{'─'*58}"
        )

        self.get_logger().info(
            scan_block + scores_block + det_block + summary_block
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _detections_to_pose_array(self, dets_xy, dets_cls):
        pose_array = PoseArray()
        for d_xy, d_cls in zip(dets_xy, dets_cls):
            p = Pose()
            p.position.x = float(d_xy[0])
            p.position.y = float(d_xy[1])
            p.position.z = 0.0
            pose_array.poses.append(p)
        return pose_array

    # ─────────────────────────────────────────────────────────────────────────
    def _detections_to_rviz_marker(self, dets_xy, dets_cls):
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns     = "dr_spaam_ros2"
        msg.id     = 0
        msg.type   = Marker.LINE_LIST

        msg.pose.orientation.w = 1.0
        msg.scale.x = 0.03
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0

        r   = 0.4
        ang = np.linspace(0, 2 * np.pi, 20)
        xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

        for d_xy, _ in zip(dets_xy, dets_cls):
            for i in range(len(xy_offsets) - 1):
                p0 = Point()
                p0.x = float(d_xy[0] + xy_offsets[i,     0])
                p0.y = float(d_xy[1] + xy_offsets[i,     1])
                msg.points.append(p0)

                p1 = Point()
                p1.x = float(d_xy[0] + xy_offsets[i + 1, 0])
                p1.y = float(d_xy[1] + xy_offsets[i + 1, 1])
                msg.points.append(p1)

        return msg


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    try:
        node = DrSpaamROS2()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error starting node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
