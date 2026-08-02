"""

DR-SPAAM detector + no-TF KF tracker, for isolating KF tuning from TF/odom pipeline

Leaser Scan -> scan_topic -> dr_spaam_ros2_node -> detections_topic -> dr_spaam_tracker_no_tf_node -> tracks_topic
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Point, Pose, PoseArray, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from proactive_nav_msgs.msg import Track as TrackMsg, TrackArray

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .dr_spaam_tracker_node import KalmanFilter, Track  # noqa: F401  (KalmanFilter used indirectly via Track)


class DrSpaamTrackerNoTfNode(Node):
    """Multi-pedestrian KF tracker with no TF/odom dependency — tracks in the sensor frame."""

    def __init__(self):
        super().__init__("dr_spaam_tracker_no_tf_node")

        self.declare_parameter("detections_topic", "/dr_spaam_ros2_node/detections")
        self.declare_parameter("tracks_topic", "~/tracks")
        self.declare_parameter("markers_topic", "~/markers")
        self.declare_parameter("track_array_topic", "~/track_array")

        self.declare_parameter("association_threshold", 1.0)
        self.declare_parameter("max_lost_frames", 10)

        self.declare_parameter("std_a_x", 1.0)
        self.declare_parameter("std_a_y", 1.0)
        self.declare_parameter("r_laser", 0.20)
        self.declare_parameter("prediction_horizon", 1.5)
        self.declare_parameter("confirm_frames", 5)

        self.declare_parameter("static_speed_threshold", 0.15)
        self.declare_parameter("static_frames_required", 15)

        self.declare_parameter("lidar_radius_m", 1.0)

        self.dets_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.tracks_topic = self.get_parameter("tracks_topic").get_parameter_value().string_value
        self.markers_topic = self.get_parameter("markers_topic").get_parameter_value().string_value
        self.track_array_topic = self.get_parameter("track_array_topic").get_parameter_value().string_value

        self.association_thresh = self.get_parameter("association_threshold").get_parameter_value().double_value
        self.max_lost_frames = self.get_parameter("max_lost_frames").get_parameter_value().integer_value

        self.std_a_x = self.get_parameter("std_a_x").get_parameter_value().double_value
        self.std_a_y = self.get_parameter("std_a_y").get_parameter_value().double_value
        self.r_laser = self.get_parameter("r_laser").get_parameter_value().double_value
        self.prediction_horizon = self.get_parameter("prediction_horizon").get_parameter_value().double_value
        self.confirm_frames = self.get_parameter("confirm_frames").get_parameter_value().integer_value
        self.static_speed_thresh = self.get_parameter("static_speed_threshold").get_parameter_value().double_value
        self.static_frames_req = self.get_parameter("static_frames_required").get_parameter_value().integer_value
        self.lidar_radius_m = self.get_parameter("lidar_radius_m").get_parameter_value().double_value

        self.tracks = []
        self.next_track_id = 1
        self.last_time = None
        self.static_blacklist = []   # [(x, y, expiry_ros_time), ...]

        self.dets_sub = self.create_subscription(
            PoseArray, self.dets_topic, self._dets_callback, 10
        )
        self.tracks_pub = self.create_publisher(PoseArray, self.tracks_topic, 10)
        self.track_array_pub = self.create_publisher(TrackArray, self.track_array_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║   DR-SPAAM Tracker (no TF/odom, sensor   ║\n"
            f"  ║   frame only — stationary robot use)     ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Subscribed to    : {self.dets_topic}\n"
            f"  Publishing tracks: {self.tracks_topic}\n"
            f"  Publishing RViz  : {self.markers_topic}\n"
            f"  Assoc. Threshold : {self.association_thresh}m\n"
            f"  Max Lost Frames  : {self.max_lost_frames}\n"
            f"  Confirm Frames   : {self.confirm_frames}\n"
            f"  Pred. Horizon    : {self.prediction_horizon}s\n"
            f"  Static Filter    : speed<{self.static_speed_thresh}m/s for {self.static_frames_req} frames\n"
            f"  Hungarian Solver : {'Scipy Hungarian' if HAS_SCIPY else 'Greedy Matching'}\n"
            f"  WARNING: no odom transform — only correct while the robot is stationary."
        )

    def _dets_callback(self, msg: PoseArray):
        try:
            self._process_detections(msg)
        except Exception as e:
            self.get_logger().error(f"[TRACKER-NO-TF] Exception in callback: {e}")

    def _process_detections(self, msg: PoseArray):
        # dt from the sensor's own timestamp, same reasoning as dr_spaam_tracker_node:
        # arrival-time jitter (inference/network) must not leak into the velocity estimate.
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if stamp.nanoseconds == 0:
            stamp = self.get_clock().now()

        if self.last_time is None:
            self.last_time = stamp
            return

        dt = (stamp - self.last_time).nanoseconds / 1e9
        self.last_time = stamp
        if dt <= 0.0 or dt > 2.0:
            dt = 0.1

        # No TF lookup, no world transform — detections are used exactly as received,
        # in the sensor's own frame (msg.header.frame_id, e.g. "base_laser").
        detections = [(p.position.x, p.position.y) for p in msg.poses]

        for track in self.tracks:
            track.predict(dt)

        matched_detections = set()
        matched_tracks = set()
        associations = []

        if len(self.tracks) > 0 and len(detections) > 0:
            if HAS_SCIPY:
                cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
                for t_idx, track in enumerate(self.tracks):
                    for d_idx, det in enumerate(detections):
                        cost_matrix[t_idx, d_idx] = np.hypot(
                            track.position[0] - det[0], track.position[1] - det[1]
                        )
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < self.association_thresh:
                        associations.append((r, c))
                        matched_tracks.add(r)
                        matched_detections.add(c)
            else:
                dists = []
                for t_idx, track in enumerate(self.tracks):
                    for d_idx, det in enumerate(detections):
                        dist = np.hypot(track.position[0] - det[0], track.position[1] - det[1])
                        if dist < self.association_thresh:
                            dists.append((dist, t_idx, d_idx))
                dists.sort(key=lambda item: item[0])
                for dist, t_idx, d_idx in dists:
                    if t_idx not in matched_tracks and d_idx not in matched_detections:
                        associations.append((t_idx, d_idx))
                        matched_tracks.add(t_idx)
                        matched_detections.add(d_idx)

        for t_idx, d_idx in associations:
            self.tracks[t_idx].update(detections[d_idx])
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.lost_count += 1

        now_time = self.get_clock().now()
        self.static_blacklist = [
            (bx, by, exp) for (bx, by, exp) in self.static_blacklist if now_time < exp
        ]
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                is_blacklisted = any(
                    np.hypot(det[0] - bx, det[1] - by) < self.association_thresh
                    for (bx, by, _) in self.static_blacklist
                )
                if not is_blacklisted:
                    new_track = Track(
                        self.next_track_id, det, self.std_a_x, self.std_a_y, self.r_laser, self.confirm_frames
                    )
                    self.tracks.append(new_track)
                    self.next_track_id += 1

        active_tracks = []
        for track in self.tracks:
            track.check_static(self.static_speed_thresh, self.static_frames_req)
            if track.lost_count > self.max_lost_frames:
                self.get_logger().info(f"[TRACKER-NO-TF] Deleting track ID {track.id} (lost for too long)")
            elif track.is_static:
                self.get_logger().info(
                    f"[TRACKER-NO-TF] Removing static object ID {track.id} — blacklisting position for 5s"
                )
                expiry = self.get_clock().now() + rclpy.duration.Duration(seconds=5.0)
                self.static_blacklist.append((track.position[0], track.position[1], expiry))
            else:
                active_tracks.append(track)
        self.tracks = active_tracks

        # Published directly in the sensor's own frame — no world_header, since there is
        # no transform. header.frame_id is whatever the incoming detections used.
        out_header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)

        tracks_msg = PoseArray()
        tracks_msg.header = out_header
        track_array = TrackArray()
        track_array.header = out_header
        for track in self.tracks:
            if track.state == "ACTIVE":
                p = Pose()
                p.position.x = track.position[0]
                p.position.y = track.position[1]
                p.position.z = 0.0
                tracks_msg.poses.append(p)

                vx, vy = track.velocity
                t = TrackMsg()
                t.id = int(track.id)
                t.x, t.y = float(track.position[0]), float(track.position[1])
                t.vx, t.vy = float(vx), float(vy)
                t.speed = float(np.hypot(vx, vy))
                t.heading = float(np.arctan2(vy, vx))
                t.moving = bool(not track.is_static)
                track_array.tracks.append(t)

        self.tracks_pub.publish(tracks_msg)
        self.track_array_pub.publish(track_array)
        self._publish_rviz_markers(out_header)

    def _publish_rviz_markers(self, header):
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.markers_pub.publish(marker_array)

        marker_array = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()

        # No robot-position reference circle here — without TF there is no "robot position
        # in this frame" concept; the sensor origin (0, 0) in its own frame is the robot.
        circle = Marker()
        circle.header = header
        circle.ns = "lidar_range_circle"
        circle.id = 0
        circle.type = Marker.LINE_STRIP
        circle.action = Marker.ADD
        circle.scale.x = 0.02
        circle.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.6)
        num_segments = 36
        circle.points = [
            Point(
                x=self.lidar_radius_m * np.cos(theta),
                y=self.lidar_radius_m * np.sin(theta),
                z=0.02,
            )
            for theta in np.linspace(0, 2 * np.pi, num_segments + 1)
        ]
        circle.lifetime = lifetime
        marker_array.markers.append(circle)

        for track in self.tracks:
            if track.state != "ACTIVE":
                continue

            vx, vy = track.prediction_velocity
            v_mag = np.hypot(vx, vy)

            cylinder = Marker()
            cylinder.header = header
            cylinder.ns = "people_cylinders"
            cylinder.id = track.id
            cylinder.type = Marker.CYLINDER
            cylinder.action = Marker.ADD
            cylinder.pose.position.x = track.position[0]
            cylinder.pose.position.y = track.position[1]
            cylinder.pose.position.z = 0.9
            cylinder.scale = Vector3(x=0.4, y=0.4, z=1.8)
            cylinder.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.7)
            cylinder.lifetime = lifetime
            marker_array.markers.append(cylinder)

            if v_mag > 0.1:
                arrow = Marker()
                arrow.header = header
                arrow.ns = "velocity_arrows"
                arrow.id = track.id
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [
                    Point(x=track.position[0], y=track.position[1], z=0.1),
                    Point(x=track.position[0] + vx, y=track.position[1] + vy, z=0.1),
                ]
                arrow.scale = Vector3(x=0.05, y=0.1, z=0.1)
                arrow.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
                arrow.lifetime = lifetime
                marker_array.markers.append(arrow)

            future_x = track.position[0] + vx * self.prediction_horizon
            future_y = track.position[1] + vy * self.prediction_horizon

            pred_sphere = Marker()
            pred_sphere.header = header
            pred_sphere.ns = "predicted_positions"
            pred_sphere.id = track.id
            pred_sphere.type = Marker.SPHERE
            pred_sphere.action = Marker.ADD
            pred_sphere.pose.position.x = future_x
            pred_sphere.pose.position.y = future_y
            pred_sphere.pose.position.z = 0.5
            pred_sphere.scale = Vector3(x=0.35, y=0.35, z=0.35)
            pred_sphere.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.5)
            pred_sphere.lifetime = lifetime
            marker_array.markers.append(pred_sphere)

            pred_path = Marker()
            pred_path.header = header
            pred_path.ns = "prediction_paths"
            pred_path.id = track.id
            pred_path.type = Marker.LINE_STRIP
            pred_path.action = Marker.ADD
            pred_path.scale.x = 0.025
            pred_path.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.6)
            pred_path.points = [
                Point(x=track.position[0], y=track.position[1], z=0.1),
                Point(x=future_x, y=future_y, z=0.1),
            ]
            pred_path.lifetime = lifetime
            marker_array.markers.append(pred_path)

            text = Marker()
            text.header = header
            text.ns = "people_labels"
            text.id = track.id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = track.position[0]
            text.pose.position.y = track.position[1]
            text.pose.position.z = 2.0
            text.scale.z = 0.25
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = f"ID: {track.id} | V: {v_mag:.2f} m/s"
            text.lifetime = lifetime
            marker_array.markers.append(text)

        if len(marker_array.markers) > 0:
            self.markers_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DrSpaamTrackerNoTfNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error starting no-TF tracker node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
