import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

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
        # Set to 360 to use the full scan. Set to 270 to remove the rear 90°,
        # which is where the robot's own body sits and is a steady source of
        # false positives if fed to the detector.
        self.declare_parameter("scan_fov_deg", 270.0)
        # Cap detections beyond this distance (metres). Overrides the sensor's hardware max.
        # Set to -1.0 to use the sensor's native range_max.
        self.declare_parameter("max_range", -1.0)
        # Fixed frame detections are transformed into before publishing. base_laser (the
        # scan's own frame_id) moves and rotates with the robot, so a stationary object
        # picks up apparent velocity from the robot's own motion whenever dr_spaam_tracker_node
        # runs its Kalman filter directly on sensor-frame positions. odom is not itself
        # perfectly drift-free, but unlike base_laser it doesn't rotate with the chassis,
        # which is what was corrupting the tracker's velocity/prediction output.
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("tf_timeout_s", 0.1)

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
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.tf_timeout_s = self.get_parameter("tf_timeout_s").get_parameter_value().double_value
        # Half-angle in radians defining the keep window: [-half_fov, +half_fov]
        self._half_fov_rad = np.deg2rad(self.scan_fov_deg / 2.0)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

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

        # Re-order by angle. The scan sweeps 0..2pi, so after normalizing to [-pi, pi]
        # the array runs 0..+180 then wraps to -180..0 — it is not monotonic. The mask
        # therefore keeps a block at the START (0..+135) and another at the END
        # (-135..0), leaving the kept array discontinuous across its middle and making
        # (last - first) about zero rather than the true 270 span. Unsorted, that fed
        # DR-SPAAM a scan that jumps +135 -> -135 mid-array and configured the detector
        # with a 0 degree FOV. Sorting restores a single contiguous -135..+135 sweep.
        order = np.argsort(scan_phi_cropped)
        scan_phi_cropped = scan_phi_cropped[order]
        raw_ranges_cropped = raw_ranges_cropped[order]

        # ── Configure detector FOV on first call (use cropped FOV) ───────────
        if not self._detector.is_ready():
            cropped_fov_deg = (
                np.rad2deg(scan_phi_cropped[-1] - scan_phi_cropped[0])
                if len(scan_phi_cropped) > 1 else self.scan_fov_deg
            )
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

        # Apply confidence threshold filter
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]

        # Transform into the fixed world frame before publishing. dr_spaam_tracker_node's
        # Kalman filter derives velocity from (position delta / dt); positions still in
        # base_laser would fold the robot's own motion into that velocity, producing wrong
        # "moving" predictions for stationary objects whenever the robot turns or drives.
        world_xy = self._transform_to_world(dets_xy, msg.header)
        if world_xy is None:
            return  # TF not ready yet — drop this scan's detections rather than
            # publish them in the wrong frame silently.

        # Convert to geometry_msgs/PoseArray
        dets_msg = self.detections_to_pose_array(world_xy, dets_cls)
        dets_msg.header.stamp = msg.header.stamp
        dets_msg.header.frame_id = self.world_frame
        self._dets_pub.publish(dets_msg)

        # Convert to visualization_msgs/Marker
        rviz_msg = self.detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)

    def _transform_to_world(self, dets_xy, header):
        """Transform Nx2 sensor-frame points into self.world_frame using the TF tree.

        Applied as a single 2D rigid transform (yaw + translation) rather than per-point
        PoseStamped transforms — the scan frame is planar (z=0) and every point in one
        scan shares the same transform, so there is nothing per-point TF lookup would add.
        """
        try:
            # odom/tf reach this machine over raw ROS2 DDS on the same WiFi link the
            # scan bridge was deliberately built to avoid (see grpc_lidar_server_node.py) —
            # in practice the newest TF sample tends to sit ~1s behind the gRPC-delivered
            # scan's own stamp, so an exact-stamp lookup would fail almost every time.
            # Time() (zero) asks for the latest transform tf2 has, not one at this exact
            # instant. For a lawnmower's walking-pace speeds, a pose up to ~1s old is
            # still far more correct than skipping the transform entirely.
            tf = self._tf_buffer.lookup_transform(
                self.world_frame,
                header.frame_id,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_s),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF {header.frame_id} -> {self.world_frame} unavailable ({e}); "
                f"dropping this scan's detections",
                throttle_duration_sec=5.0,
            )
            return None

        if len(dets_xy) == 0:
            return dets_xy

        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        dets_xy = np.asarray(dets_xy)
        world_xy = np.empty_like(dets_xy)
        world_xy[:, 0] = t.x + cos_yaw * dets_xy[:, 0] - sin_yaw * dets_xy[:, 1]
        world_xy[:, 1] = t.y + sin_yaw * dets_xy[:, 0] + cos_yaw * dets_xy[:, 1]
        return world_xy

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