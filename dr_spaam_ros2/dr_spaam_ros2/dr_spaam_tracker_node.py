import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import Point, Pose, PoseArray, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# Try importing scipy for Hungarian algorithm, fallback to greedy association if not present
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class KalmanFilter:
    """Standard Kalman Filter for a 2D Constant Velocity (CV) model.

    State vector: [px, py, vx, vy]
    Measurement: [px, py] from LiDAR detections

    Args:
        x_init: Initial state [px, py, vx, vy]
        std_a_x: Process acceleration noise std dev in X (m/s^2)
        std_a_y: Process acceleration noise std dev in Y (m/s^2)
        r_laser: Measurement noise std dev for LiDAR position (m)
    """

    def __init__(self, x_init, std_a_x=1.5, std_a_y=1.5, r_laser=0.15):
        # Use float64 for numerical stability (avoids SIGABRT from singular matrices in float32)
        self.x = np.array(x_init, dtype=np.float64).reshape(4, 1)

        # State covariance matrix P
        # Diagonal: low uncertainty for position (measured), high for velocity (unknown at init)
        self.P = np.array([
            [1.0, 0.0, 0.0,  0.0],
            [0.0, 1.0, 0.0,  0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0]
        ], dtype=np.float64)

        # State transition matrix F (updated per dt in predict())
        self.F = np.eye(4, dtype=np.float64)

        # Lidar Measurement matrix H (linear mapping of px, py)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Measurement noise covariance matrix R (Lidar)
        self.R_laser = np.eye(2, dtype=np.float64) * (r_laser ** 2)

        # Process noise parameters — MUST be > 0 even in Constant Velocity model.
        # They represent the budget of unmodelled velocity changes per second.
        # Setting these to 0 makes Q=0 → velocity estimate NEVER updates.
        self.std_a_x = std_a_x
        self.std_a_y = std_a_y

        # Adaptive Q scale — boosted on maneuver detection (large NIS), then decays.
        # Allows fast velocity re-estimation after sudden direction changes.
        self._q_scale = 1.0

    def predict(self, dt):
        """Predict the next state and covariance based on dt."""
        # 1. Update transition matrix F with dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # 2. Compute process noise covariance matrix Q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        Q = np.zeros((4, 4), dtype=np.float64)
        noise_ax = self.std_a_x ** 2
        noise_ay = self.std_a_y ** 2

        Q[0, 0] = (dt4 / 4.0) * noise_ax
        Q[0, 2] = (dt3 / 2.0) * noise_ax
        Q[1, 1] = (dt4 / 4.0) * noise_ay
        Q[1, 3] = (dt3 / 2.0) * noise_ay
        Q[2, 0] = (dt3 / 2.0) * noise_ax
        Q[2, 2] = dt2 * noise_ax
        Q[3, 1] = (dt3 / 2.0) * noise_ay
        Q[3, 3] = dt2 * noise_ay

        # Apply adaptive scale and decay toward nominal.
        # After a maneuver (NIS spike), Q is inflated for ~1–2 s so velocity re-estimates fast.
        Q *= self._q_scale
        self._q_scale = max(1.0, self._q_scale * 0.85)

        # 3. Perform prediction
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z):
        """Standard Kalman Filter update for linear Lidar measurements (px, py)."""
        z = np.array(z, dtype=np.float64).reshape(2, 1)

        # Measurement residual
        y = z - self.H @ self.x

        # Project system uncertainty into measurement space
        S = self.H @ self.P @ self.H.T + self.R_laser

        # Normalized Innovation Squared (NIS): chi2-distributed with dof=2 under correct model.
        # Expected mean ≈ 2. NIS >> 2 indicates a maneuver — boost Q for the next predict step
        # so velocity adapts quickly instead of lagging for multiple frames.
        try:
            nis = (y.T @ np.linalg.solve(S, y)).item()  # (1,1) array → scalar
            if nis > 9.0:  # > 99th percentile of chi2(2): maneuver detected
                self._q_scale = min(15.0, nis / 2.0)
        except (np.linalg.LinAlgError, ValueError):
            pass

        # Compute Kalman gain using solve (numerically safer than inv)
        # K = P @ H^T @ S^-1  =>  S^T @ K^T = (P @ H^T)^T  =>  use solve
        try:
            K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        except np.linalg.LinAlgError:
            # If S is singular, skip this update
            return

        # Update state
        self.x = self.x + K @ y

        # Update covariance using Joseph stabilized form:
        # P = (I - KH) @ P @ (I - KH)^T + K @ R @ K^T
        # This keeps P symmetric and positive definite across all iterations,
        # preventing numerical decay that would freeze the velocity estimate.
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_laser @ K.T



class Track:
    """A tracked target representing a single pedestrian."""

    # EMA smoothing factor for prediction velocity (0 = frozen, 1 = raw KF velocity).
    # Smooths the prediction arrow direction without touching position tracking.
    _VEL_ALPHA = 0.35

    def __init__(self, track_id, position, std_a_x, std_a_y, r_laser):
        self.id = track_id
        # Initialize state with position and zero velocity
        x_init = [position[0], position[1], 0.0, 0.0]
        self.kf = KalmanFilter(x_init, std_a_x, std_a_y, r_laser)

        self.age = 1
        self.lost_count = 0
        self.static_count = 0       # Consecutive frames where speed < threshold
        self.is_static = False       # True if classified as static object
        self.state = "TENTATIVE"    # "TENTATIVE", "ACTIVE"

        self._smooth_vx = 0.0
        self._smooth_vy = 0.0

    def predict(self, dt):
        self.kf.predict(dt)

    def update(self, position):
        self.kf.update(position)
        self.lost_count = 0
        self.age += 1
        # Require 5 consecutive matches before publishing — filters out DR-SPAAM spurious blips
        # that appear for 1–3 frames then vanish (walls, glass, passing objects).
        if self.state == "TENTATIVE" and self.age >= 5:
            self.state = "ACTIVE"

        # Update smoothed velocity for prediction (EMA — does not affect KF state)
        vx, vy = self.velocity
        self._smooth_vx = self._VEL_ALPHA * vx + (1.0 - self._VEL_ALPHA) * self._smooth_vx
        self._smooth_vy = self._VEL_ALPHA * vy + (1.0 - self._VEL_ALPHA) * self._smooth_vy

    def check_static(self, speed_threshold, static_frames_required):
        """Check if this track is a static object based on velocity.
        If speed stays below threshold for N consecutive frames, mark as static."""
        speed = self.speed
        if speed < speed_threshold:
            self.static_count += 1
        else:
            # Person started moving — reset static counter
            self.static_count = 0
            self.is_static = False

        # Only classify as static after the track has had enough time to
        # converge its velocity estimate (age > static_frames_required)
        if self.static_count >= static_frames_required and self.age > static_frames_required:
            self.is_static = True

    @property
    def position(self):
        return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))

    @property
    def velocity(self):
        return (float(self.kf.x[2, 0]), float(self.kf.x[3, 0]))

    @property
    def speed(self):
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def prediction_velocity(self):
        """EMA-smoothed velocity used only for future-position prediction markers."""
        return (self._smooth_vx, self._smooth_vy)


class DrSpaamTrackerNode(Node):
    """ROS2 Node for Multi-Pedestrian Tracking using Kalman Filters."""

    def __init__(self):
        super().__init__("dr_spaam_tracker_node")

        # ── Declare and Get Parameters ──────────────────────────────────────
        self.declare_parameter("detections_topic", "/dr_spaam_ros2_node/detections")
        self.declare_parameter("tracks_topic", "~/tracks")
        self.declare_parameter("markers_topic", "~/markers")

        # Association and Track management limits
        self.declare_parameter("association_threshold", 1.0)  # Max matching dist (meters)
        self.declare_parameter("max_lost_frames", 10)         # Max frames to keep lost track

        # EKF/KF Process and Measurement parameters
        self.declare_parameter("std_a_x", 1.5)                 # Process acceleration noise X
        self.declare_parameter("std_a_y", 1.5)                 # Process acceleration noise Y
        self.declare_parameter("r_laser", 0.20)                # Measurement noise (std dev)
        self.declare_parameter("prediction_horizon", 1.5)      # Prediction time horizon (seconds)

        # Static object filter parameters
        self.declare_parameter("static_speed_threshold", 0.15)  # Speed below this (m/s) is considered static
        self.declare_parameter("static_frames_required", 15)    # Consecutive static frames before removal

        # Extract values
        self.dets_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.tracks_topic = self.get_parameter("tracks_topic").get_parameter_value().string_value
        self.markers_topic = self.get_parameter("markers_topic").get_parameter_value().string_value

        self.association_thresh = self.get_parameter("association_threshold").get_parameter_value().double_value
        self.max_lost_frames = self.get_parameter("max_lost_frames").get_parameter_value().integer_value

        self.std_a_x = self.get_parameter("std_a_x").get_parameter_value().double_value
        self.std_a_y = self.get_parameter("std_a_y").get_parameter_value().double_value
        self.r_laser = self.get_parameter("r_laser").get_parameter_value().double_value
        self.prediction_horizon = self.get_parameter("prediction_horizon").get_parameter_value().double_value
        self.static_speed_thresh = self.get_parameter("static_speed_threshold").get_parameter_value().double_value
        self.static_frames_req = self.get_parameter("static_frames_required").get_parameter_value().integer_value

        # ── State Initialization ─────────────────────────────────────────────
        self.tracks = []
        self.next_track_id = 1
        self.last_time = None
        self._stale_msg_count = 0
        # Static blacklist: list of (x, y, expiry_time) for positions of removed static objects.
        # New detections too close to a blacklisted position are rejected (not spawned as tracks).
        self.static_blacklist = []   # [(x, y, expiry_ros_time), ...]

        # ── Subscriptions & Publishers ───────────────────────────────────────
        self.dets_sub = self.create_subscription(
            PoseArray, self.dets_topic, self._dets_callback, 10
        )
        self.tracks_pub = self.create_publisher(PoseArray, self.tracks_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║      DR-SPAAM Multi-Target Tracker       ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  Subscribed to    : {self.dets_topic}\n"
            f"  Publishing tracks: {self.tracks_topic}\n"
            f"  Publishing RViz  : {self.markers_topic}\n"
            f"  Assoc. Threshold : {self.association_thresh}m\n"
            f"  Max Lost Frames  : {self.max_lost_frames}\n"
            f"  Pred. Horizon    : {self.prediction_horizon}s\n"
            f"  Static Filter    : speed<{self.static_speed_thresh}m/s for {self.static_frames_req} frames\n"
            f"  Hungarian Solver : {'Scipy Hungarian' if HAS_SCIPY else 'Greedy Matching'}"
        )

    def _dets_callback(self, msg: PoseArray):
        try:
            self._process_detections(msg)
        except Exception as e:
            self.get_logger().error(f"[TRACKER] Exception in callback: {e}")

    def _process_detections(self, msg: PoseArray):
        # 1. Compute dt from the SENSOR timestamp, not arrival time.
        # Velocity is effectively (position delta / dt), so dt must measure the interval
        # the position delta actually happened over. Arrival time does not: if the same
        # scan is delivered several times (duplicate publishers, a /scan feedback loop),
        # arrival times are milliseconds apart while the position delta still spans a
        # full scan period, so the filter infers a velocity inflated by the duplication
        # factor and the track shoots forward. Sensor stamps make duplicates identical
        # (dt == 0), which the guard below then drops instead of believing.
        stamp = Time.from_msg(msg.header.stamp)
        if stamp.nanoseconds == 0:
            # Upstream published an unset stamp — fall back to arrival time.
            stamp = self.get_clock().now()

        if self.last_time is None:
            self.last_time = stamp
            return

        dt = (stamp - self.last_time).nanoseconds / 1e9

        # Avoid dt anomalies
        # A non-positive dt means this message carries a stamp we have already seen
        # (a duplicate) or one older than the last (out of order). Substituting a
        # nominal 0.1 s here — the previous behaviour — makes the filter advance a full
        # prediction step for a scan that represents no elapsed time. Under a duplicate
        # storm that compounds: N duplicates per second each advance 0.1 s, so the CV
        # model extrapolates N/10 times faster than real time and tracks visibly race
        # ahead of the person the moment velocity becomes non-zero. Drop the message
        # instead — re-predicting on a repeated observation cannot add information.
        if dt <= 0.0:
            self._stale_msg_count += 1
            if self._stale_msg_count % 50 == 1:
                self.get_logger().warn(
                    f"Dropped {self._stale_msg_count} duplicate/out-of-order detection "
                    f"message(s) (dt={dt:.4f}s). Check for multiple publishers on the "
                    f"scan topic — a /scan feedback loop produces exactly this."
                )
            return

        # A large gap is a real stall (node restart, dropped link) rather than a
        # duplicate; clamp so Q stays sane instead of exploding on a huge dt.
        if dt > 2.0:
            dt = 0.1

        # Only advance the reference once the message is accepted. Updating it before
        # the guards would let a rejected out-of-order stamp become the new baseline,
        # making the next (correctly ordered) message compute an inflated dt.
        self.last_time = stamp

        # 2. Extract detection coordinates
        detections = []
        for pose in msg.poses:
            detections.append((pose.position.x, pose.position.y))

        # 3. Kalman Filter PREDICT step for all active tracks
        for track in self.tracks:
            track.predict(dt)

        # 4. DATA ASSOCIATION
        matched_detections = set()
        matched_tracks = set()
        associations = []

        if len(self.tracks) > 0 and len(detections) > 0:

            if HAS_SCIPY:
                # Compute cost matrix (Euclidean distances)
                cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
                for t_idx, track in enumerate(self.tracks):
                    for d_idx, det in enumerate(detections):
                        dist = np.hypot(track.position[0] - det[0], track.position[1] - det[1])
                        cost_matrix[t_idx, d_idx] = dist

                # Run Hungarian matching
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < self.association_thresh:
                        associations.append((r, c))
                        matched_tracks.add(r)
                        matched_detections.add(c)
            else:
                # Fallback: Greedy Nearest Neighbor matching
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

        # Apply associations (Kalman UPDATE step)
        for t_idx, d_idx in associations:
            self.tracks[t_idx].update(detections[d_idx])
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

        # 5. TRACK LIFECYCLE MANAGEMENT
        # Unmatched tracks are missed/lost
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.lost_count += 1

        # Unmatched detections spawn new tentative tracks.
        # Skip detections that fall inside the static blacklist zone.
        now_time = self.get_clock().now()
        # Expire old blacklist entries
        self.static_blacklist = [
            (bx, by, exp) for (bx, by, exp) in self.static_blacklist
            if now_time < exp
        ]
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                # Check if this detection is near a blacklisted static position
                is_blacklisted = any(
                    np.hypot(det[0] - bx, det[1] - by) < self.association_thresh
                    for (bx, by, _) in self.static_blacklist
                )
                if not is_blacklisted:
                    new_track = Track(self.next_track_id, det, self.std_a_x, self.std_a_y, self.r_laser)
                    self.tracks.append(new_track)
                    self.next_track_id += 1

        # Prune dead tracks AND static objects
        active_tracks = []
        for track in self.tracks:
            # Check if track is static (false positive from wall/glass/furniture)
            track.check_static(self.static_speed_thresh, self.static_frames_req)

            if track.lost_count > self.max_lost_frames:
                self.get_logger().info(f"[TRACKER] Deleting track ID {track.id} (lost for too long)")
            elif track.is_static:
                self.get_logger().info(
                    f"[TRACKER] Removing static object ID {track.id} "
                    f"(speed={track.speed:.3f}m/s < {self.static_speed_thresh}m/s "
                    f"for {track.static_count} frames) — blacklisting position for 5s"
                )
                # Add to blacklist so the same position won't spawn a new track for 5 seconds
                expiry = self.get_clock().now() + rclpy.duration.Duration(seconds=5.0)
                self.static_blacklist.append((track.position[0], track.position[1], expiry))
            else:
                active_tracks.append(track)
        self.tracks = active_tracks

        # 6. PUBLISH TRACKED POSES (only moving, active tracks)
        tracks_msg = PoseArray()
        tracks_msg.header = msg.header
        for track in self.tracks:
            if track.state == "ACTIVE":
                p = Pose()
                p.position.x = track.position[0]
                p.position.y = track.position[1]
                p.position.z = 0.0
                tracks_msg.poses.append(p)
        self.tracks_pub.publish(tracks_msg)

        # 7. PUBLISH RVIZ MARKERS
        self._publish_rviz_markers(msg.header)

    def _publish_rviz_markers(self, header):
        marker_array = MarkerArray()

        # Clear old markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.markers_pub.publish(marker_array)

        marker_array = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()

        for track in self.tracks:
            if track.state != "ACTIVE":
                continue

            vx, vy = track.prediction_velocity
            v_mag = np.hypot(vx, vy)

            # ── 1. Cylinder representing the person ──────────────────────────
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

            # ── 2. Velocity arrow (Yellow) ───────────────────────────────────
            if v_mag > 0.1:
                arrow = Marker()
                arrow.header = header
                arrow.ns = "velocity_arrows"
                arrow.id = track.id
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                p_start = Point(x=track.position[0], y=track.position[1], z=0.1)
                p_end = Point(x=track.position[0] + vx, y=track.position[1] + vy, z=0.1)
                arrow.points = [p_start, p_end]
                arrow.scale = Vector3(x=0.05, y=0.1, z=0.1)
                arrow.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
                arrow.lifetime = lifetime
                marker_array.markers.append(arrow)

            # ── 3. NEXT STATE PREDICTION (Green) ─────────────────────────────
            # Calculate predicted future position using CV model
            future_x = track.position[0] + vx * self.prediction_horizon
            future_y = track.position[1] + vy * self.prediction_horizon

            # 3a. Predicted Position Sphere (semi-transparent green ghost)
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

            # 3b. Prediction Path (Green line from current to future)
            pred_path = Marker()
            pred_path.header = header
            pred_path.ns = "prediction_paths"
            pred_path.id = track.id
            pred_path.type = Marker.LINE_STRIP
            pred_path.action = Marker.ADD
            pred_path.scale.x = 0.025
            pred_path.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.6)
            pt_current = Point(x=track.position[0], y=track.position[1], z=0.1)
            pt_future = Point(x=future_x, y=future_y, z=0.1)
            pred_path.points = [pt_current, pt_future]
            pred_path.lifetime = lifetime
            marker_array.markers.append(pred_path)

            # ── 4. Text label (White) ────────────────────────────────────────
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
        node = DrSpaamTrackerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error starting tracker node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()