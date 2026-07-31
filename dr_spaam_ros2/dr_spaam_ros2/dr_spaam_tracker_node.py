import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import Point, Pose, PoseArray, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from proactive_nav_msgs.msg import Track as TrackMsg, TrackArray
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

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

    def __init__(self, track_id, position, std_a_x, std_a_y, r_laser, confirm_frames):
        self.id = track_id
        # Initialize state with position and zero velocity
        x_init = [position[0], position[1], 0.0, 0.0]
        self.kf = KalmanFilter(x_init, std_a_x, std_a_y, r_laser)

        # Consecutive matched updates required before this track is published as ACTIVE.
        self.confirm_frames = confirm_frames
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
        # Require confirm_frames consecutive matches before publishing — filters out DR-SPAAM
        # spurious blips that appear for a few frames then vanish (walls, glass, passing objects).
        if self.state == "TENTATIVE" and self.age >= self.confirm_frames:
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
        self.declare_parameter("track_array_topic", "~/track_array")

        # Association and Track management limits
        self.declare_parameter("association_threshold", 1.0)  # Max matching dist (meters)
        self.declare_parameter("max_lost_frames", 10)         # Max frames to keep lost track

        # EKF/KF Process and Measurement parameters
        self.declare_parameter("std_a_x", 1.5)                 # Process acceleration noise X
        self.declare_parameter("std_a_y", 1.5)                 # Process acceleration noise Y
        self.declare_parameter("r_laser", 0.20)                # Measurement noise (std dev)
        self.declare_parameter("prediction_horizon", 1.5)      # Prediction time horizon (seconds)
        # Consecutive matched frames before a new track is confirmed ACTIVE and published.
        # Lower = faster to show up as tracked, but more likely to confirm a spurious blip.
        self.declare_parameter("confirm_frames", 5)

        # Static object filter parameters
        self.declare_parameter("static_speed_threshold", 0.15)  # Speed below this (m/s) is considered static
        self.declare_parameter("static_frames_required", 15)    # Consecutive static frames before removal

        # Reference circle drawn around the lidar origin (visualization only)
        self.declare_parameter("lidar_radius_m", 1.0)

        # World-fixed frame the Kalman filter actually tracks in. The constant-velocity
        # model is only physically meaningful in a frame that doesn't itself move — if
        # detections stayed in the sensor's own frame (e.g. "base_laser") once the robot
        # is driving, a stationary person would appear to move (and a moving person's
        # velocity would be wrong) purely from the robot's own motion. Detections are
        # transformed into this frame before they ever reach the KF; nothing about the
        # filter/association math changes. Requires odom->base_link (or whatever the
        # incoming scan's frame_id is) to actually be on the TF tree.
        self.declare_parameter("target_frame", "odom")
        # How long to wait for the transform to become available before giving up on a
        # given detections message. Kept short: at 10Hz a scan is stale well before 0.2s.
        self.declare_parameter("tf_timeout_s", 0.2)

        # Extract values
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
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.tf_timeout_s = self.get_parameter("tf_timeout_s").get_parameter_value().double_value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Throttle the "TF not ready" warning: at startup this is expected for the first
        # second or so (odom->base_link hasn't been broadcast yet), not worth logging once
        # per scan.
        self._last_tf_warn_time = None

        # ── State Initialization ─────────────────────────────────────────────
        self.tracks = []
        self.next_track_id = 1
        self.last_time = None
        # Static blacklist: list of (x, y, expiry_time) for positions of removed static objects.
        # New detections too close to a blacklisted position are rejected (not spawned as tracks).
        self.static_blacklist = []   # [(x, y, expiry_ros_time), ...]

        # ── Subscriptions & Publishers ───────────────────────────────────────
        self.dets_sub = self.create_subscription(
            PoseArray, self.dets_topic, self._dets_callback, 10
        )
        self.tracks_pub = self.create_publisher(PoseArray, self.tracks_topic, 10)
        # Full state (id, position, velocity, heading) for off-board consumers.
        self.track_array_pub = self.create_publisher(TrackArray, self.track_array_topic, 10)
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
            f"  Confirm Frames   : {self.confirm_frames}\n"
            f"  Pred. Horizon    : {self.prediction_horizon}s\n"
            f"  Static Filter    : speed<{self.static_speed_thresh}m/s for {self.static_frames_req} frames\n"
            f"  Hungarian Solver : {'Scipy Hungarian' if HAS_SCIPY else 'Greedy Matching'}"
        )

    def _lookup_world_transform(self, source_frame, stamp):
        """Return (tx, ty, yaw) for source_frame -> self.target_frame at stamp, or None
        if the transform isn't available yet (logged, throttled — not an error at
        startup before the first odom->base_link broadcast has arrived)."""
        if source_frame == self.target_frame:
            return (0.0, 0.0, 0.0)   # already in the target frame, nothing to do

        try:
            t = self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, stamp,
                timeout=Duration(seconds=self.tf_timeout_s),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            now = self.get_clock().now()
            if self._last_tf_warn_time is None or (now - self._last_tf_warn_time).nanoseconds > 2e9:
                self._last_tf_warn_time = now
                self.get_logger().warn(
                    f"[TRACKER] No transform {source_frame} -> {self.target_frame} yet "
                    f"({e}); dropping this scan's detections until it's available."
                )
            return None

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        qz = t.transform.rotation.z
        qw = t.transform.rotation.w
        yaw = 2.0 * np.arctan2(qz, qw)   # planar transform: qx=qy=0 for a 2D robot
        return (tx, ty, yaw)

    @staticmethod
    def _apply_planar_transform(x, y, tx, ty, yaw):
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)
        return (tx + x * cos_t - y * sin_t, ty + x * sin_t + y * cos_t)

    def _dets_callback(self, msg: PoseArray):
        try:
            self._process_detections(msg)
        except Exception as e:
            self.get_logger().error(f"[TRACKER] Exception in callback: {e}")

    def _process_detections(self, msg: PoseArray):
        # 1. Compute time step dt from the SENSOR timestamp, not arrival time.
        # dt scales F and Q, and velocity is effectively (position delta / dt), so any
        # error here propagates straight into the velocity estimate. Callback arrival
        # time includes network jitter and variable inference latency — over a WiFi
        # link that is tens of ms of noise on a ~100 ms step. The header stamp is set
        # once by the lidar driver, so successive stamps differ by the true scan
        # interval regardless of transport delay. (Any constant clock offset between
        # robot and laptop cancels in the difference.)
        stamp = Time.from_msg(msg.header.stamp)
        if stamp.nanoseconds == 0:
            # Driver published an unset stamp — fall back to arrival time.
            stamp = self.get_clock().now()

        if self.last_time is None:
            self.last_time = stamp
            return

        dt = (stamp - self.last_time).nanoseconds / 1e9
        self.last_time = stamp

        # Avoid dt anomalies
        if dt <= 0.0 or dt > 2.0:
            dt = 0.1

        # 2. Extract detection coordinates and transform into the world-fixed tracking
        # frame. Everything from here on (association, KF predict/update, publishing)
        # operates in self.target_frame, not the sensor's own frame.
        world_tf = self._lookup_world_transform(msg.header.frame_id, stamp)
        if world_tf is None:
            return
        tx, ty, yaw = world_tf

        detections = []
        for pose in msg.poses:
            wx, wy = self._apply_planar_transform(pose.position.x, pose.position.y, tx, ty, yaw)
            detections.append((wx, wy))

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
                    new_track = Track(
                        self.next_track_id, det, self.std_a_x, self.std_a_y, self.r_laser, self.confirm_frames
                    )
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
        # Same timestamp as the source scan, but frame_id is now self.target_frame —
        # everything published from here on is in world-fixed coordinates, not the
        # sensor's own frame (see the transform applied above in step 2). Built as a new
        # Header rather than mutating msg.header in place, which would silently change
        # the frame_id on the incoming message object itself.
        world_header = Header(stamp=msg.header.stamp, frame_id=self.target_frame)

        tracks_msg = PoseArray()
        tracks_msg.header = world_header
        track_array = TrackArray()
        track_array.header = world_header
        for track in self.tracks:
            if track.state == "ACTIVE":
                p = Pose()
                p.position.x = track.position[0]
                p.position.y = track.position[1]
                p.position.z = 0.0
                tracks_msg.poses.append(p)

                # Full state for consumers off-board. PoseArray carries position only,
                # so velocity — the thing a proactive planner actually needs — had no
                # way out of this node except as RViz arrows.
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

        # 7. PUBLISH RVIZ MARKERS
        # (tx, ty) is the robot's own position in target_frame — reusing the transform
        # already looked up above rather than doing a second TF lookup for the same
        # instant.
        self._publish_rviz_markers(world_header, robot_x=tx, robot_y=ty)

    def _publish_rviz_markers(self, header, robot_x, robot_y):
        marker_array = MarkerArray()

        # Clear old markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.markers_pub.publish(marker_array)

        marker_array = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()

        # ── 0. Reference circle around the lidar (i.e. the robot) ────────────
        # Centered at the robot's live position, not the origin — correct when this
        # was published in the sensor's own frame (origin = the lidar), but header is
        # now target_frame (e.g. "odom"), where the origin is just wherever the robot
        # happened to start, not where the robot is now.
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
                x=robot_x + self.lidar_radius_m * np.cos(theta),
                y=robot_y + self.lidar_radius_m * np.sin(theta),
                z=0.02,
            )
            for theta in np.linspace(0, 2 * np.pi, num_segments + 1)  # +1 closes the loop
        ]
        circle.lifetime = lifetime
        marker_array.markers.append(circle)

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