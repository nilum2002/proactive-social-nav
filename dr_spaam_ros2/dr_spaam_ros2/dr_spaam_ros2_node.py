import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker

from dr_spaam.detector import Detector


class DrSpaamROS2(Node):
    """ROS2 node to detect pedestrians using DROW3 or DR-SPAAM."""

    def __init__(self):
        super().__init__("dr_spaam_ros2")

        # Declare parameters
        self.declare_parameter("weight_file", "")
        self.declare_parameter("detector_model", "DR-SPAAM")
        self.declare_parameter("conf_thresh", 0.8)
        self.declare_parameter("stride", 1)
        self.declare_parameter("panoramic_scan", True)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("gpu", True)
        # Crop the scan to this many degrees, centered on 0° (forward).
        # Set to 360 to use the full scan. Set to 270 to remove the rear 90°.
        self.declare_parameter("scan_fov_deg", 270.0)
        # Cap detections beyond this distance (metres). Overrides the sensor's hardware max.
        # Set to -1.0 to use the sensor's native range_max.
        self.declare_parameter("max_range", -1.0)

        # Get parameters
        self.weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
        self.detector_model = self.get_parameter("detector_model").get_parameter_value().string_value
        self.conf_thresh = self.get_parameter("conf_thresh").get_parameter_value().double_value
        self.stride = self.get_parameter("stride").get_parameter_value().integer_value
        self.panoramic_scan = self.get_parameter("panoramic_scan").get_parameter_value().bool_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.use_gpu = self.get_parameter("gpu").get_parameter_value().bool_value
        self.scan_fov_deg = self.get_parameter("scan_fov_deg").get_parameter_value().double_value
        self.max_range = self.get_parameter("max_range").get_parameter_value().double_value
        # Half-angle in radians defining the keep window: [-half_fov, +half_fov]
        self._half_fov_rad = np.deg2rad(self.scan_fov_deg / 2.0)

        self.get_logger().info(f"Initializing detector '{self.detector_model}' with checkpoint: '{self.weight_file}'")

        if not self.weight_file:
            self.get_logger().error("Parameter 'weight_file' is empty! Please provide a path to a valid checkpoint.")
            raise ValueError("weight_file parameter is empty.")

        # Initialize detector
        self._detector = Detector(
            self.weight_file,
            model=self.detector_model,
            gpu=self.use_gpu,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )

        # Create Publisher
        self._dets_pub = self.create_publisher(PoseArray, "~/detections", 10)
        self._rviz_pub = self.create_publisher(Marker, "~/rviz_marker", 10)

        # Create Subscriber
        self._scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            10
        )

        self.get_logger().info(f"Node initialized. Subscribed to topic: {self.scan_topic}")

    def _scan_callback(self, msg):
        # ── Build raw scan and per-ray angle arrays ───────────────────────────
        raw_ranges = np.array(msg.ranges, dtype=np.float32)
        scan_phi = msg.angle_min + np.arange(len(raw_ranges)) * msg.angle_increment

        # ── Normalize angles to [-π, π] for symmetric FOV cropping ───────────
        scan_phi_norm = (scan_phi + np.pi) % (2 * np.pi) - np.pi

        # ── Apply FOV crop: keep only rays within ±half_fov of forward (0°) ──
        keep_mask = np.abs(scan_phi_norm) <= self._half_fov_rad
        scan_phi_cropped = scan_phi_norm[keep_mask]
        raw_ranges_cropped = raw_ranges[keep_mask]

        # ── Configure detector FOV on first call (use cropped FOV) ───────────
        if not self._detector.is_ready():
            actual_fov_deg = np.rad2deg(self.scan_fov_deg if np.deg2rad(self.scan_fov_deg) <= np.pi * 2
                                        else msg.angle_increment * len(raw_ranges))
            cropped_fov_deg = np.rad2deg(scan_phi_cropped[-1] - scan_phi_cropped[0]) if len(scan_phi_cropped) > 1 else self.scan_fov_deg
            self._detector.set_laser_fov(cropped_fov_deg)
            self.get_logger().info(
                f"FOV crop applied: full={np.rad2deg(msg.angle_increment * len(raw_ranges)):.1f}° "
                f"→ cropped={cropped_fov_deg:.1f}° "
                f"({len(raw_ranges_cropped)}/{len(raw_ranges)} rays kept)"
            )

        # ── Replace invalid ranges with background ────────────────────────────
        effective_max = self.max_range if self.max_range > 0.0 else msg.range_max
        scan = raw_ranges_cropped.copy()
        scan[scan < msg.range_min] = 29.99
        scan[scan > effective_max] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        # ── Run DR-SPAAM inference on the cropped scan ────────────────────────
        dets_xy, dets_cls, _ = self._detector(scan, scan_phi=scan_phi_cropped)

        # ── Apply confidence threshold filter ─────────────────────────────────
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]

        # ── Publish detections ────────────────────────────────────────────────
        dets_msg = self.detections_to_pose_array(dets_xy, dets_cls)
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

        rviz_msg = self.detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)

    def detections_to_pose_array(self, dets_xy, dets_cls):
        pose_array = PoseArray()
        for d_xy, d_cls in zip(dets_xy, dets_cls):
            p = Pose()
            p.position.x = float(d_xy[0])
            p.position.y = float(d_xy[1])
            p.position.z = 0.0
            pose_array.poses.append(p)
        return pose_array

    def detections_to_rviz_marker(self, dets_xy, dets_cls):
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns = "dr_spaam_ros2"
        msg.id = 0
        msg.type = Marker.LINE_LIST

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        msg.scale.x = 0.03  # Line width
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0

        # Approximating a circle around each pedestrian with 20 line segments
        r = 0.4  # Radius of 0.4 meters
        ang = np.linspace(0, 2 * np.pi, 20)
        xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

        for d_xy, d_cls in zip(dets_xy, dets_cls):
            for i in range(len(xy_offsets) - 1):
                p0 = Point()
                p0.x = float(d_xy[0] + xy_offsets[i, 0])
                p0.y = float(d_xy[1] + xy_offsets[i, 1])
                p0.z = 0.0
                msg.points.append(p0)

                p1 = Point()
                p1.x = float(d_xy[0] + xy_offsets[i + 1, 0])
                p1.y = float(d_xy[1] + xy_offsets[i + 1, 1])
                p1.z = 0.0
                msg.points.append(p1)

        return msg


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