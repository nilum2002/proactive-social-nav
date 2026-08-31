"""Multi-pedestrian Kalman tracking, decoupled from ROS and gRPC.

Ported from dr_spaam_ros2's dr_spaam_tracker_node: same constant-velocity KF,
Hungarian (or greedy) association, but driven by direct step(dt, detections)
calls instead of a topic subscription so it can run inside the gRPC servicer
thread.

Static tracks are classified (Track.is_static) but NOT suppressed: a person who
stops walking keeps their track and their id. The upstream tracker dropped them
and blacklisted their position, which is right for navigation -- a stationary
person is not a moving obstacle -- but it makes CLEAR MOT meaningless, since
every stationary ground-truth person becomes a run of misses and then an id
switch when they move off again. Consumers that want the navigation behaviour
should filter on `is_static` themselves.
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class KalmanFilter:
    """Constant-velocity 2D Kalman filter. State: [px, py, vx, vy]."""

    def __init__(self, x_init, std_a_x=1.5, std_a_y=1.5, r_laser=0.15):
        # float64 avoids SIGABRT from singular matrices that float32 can hit.
        self.x = np.array(x_init, dtype=np.float64).reshape(4, 1)

        self.P = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
        ], dtype=np.float64)

        self.F = np.eye(4, dtype=np.float64)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        self.R_laser = np.eye(2, dtype=np.float64) * (r_laser ** 2)

        self.std_a_x = std_a_x
        self.std_a_y = std_a_y

        # Adaptive Q scale, boosted on maneuver detection (large NIS), then decays.
        self._q_scale = 1.0

    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        noise_ax = self.std_a_x ** 2
        noise_ay = self.std_a_y ** 2

        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = (dt4 / 4.0) * noise_ax
        Q[0, 2] = (dt3 / 2.0) * noise_ax
        Q[1, 1] = (dt4 / 4.0) * noise_ay
        Q[1, 3] = (dt3 / 2.0) * noise_ay
        Q[2, 0] = (dt3 / 2.0) * noise_ax
        Q[2, 2] = dt2 * noise_ax
        Q[3, 1] = (dt3 / 2.0) * noise_ay
        Q[3, 3] = dt2 * noise_ay

        Q *= self._q_scale
        self._q_scale = max(1.0, self._q_scale * 0.85)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z):
        z = np.array(z, dtype=np.float64).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R_laser

        # NIS ~ chi2(2) under a correct model (mean 2); a spike means a maneuver,
        # so Q is inflated for the next predict() to let velocity re-converge fast.
        try:
            nis = (y.T @ np.linalg.solve(S, y)).item()
            if nis > 9.0:  # > 99th percentile of chi2(2)
                self._q_scale = min(15.0, nis / 2.0)
        except (np.linalg.LinAlgError, ValueError):
            pass

        try:
            K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        except np.linalg.LinAlgError:
            return

        self.x = self.x + K @ y

        # Joseph form keeps P symmetric/PD across iterations.
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_laser @ K.T


class Track:
    """A single tracked pedestrian."""

    _VEL_ALPHA = 0.35  # EMA smoothing for the reported/predicted velocity only.

    def __init__(self, track_id, position, std_a_x, std_a_y, r_laser):
        self.id = track_id
        self.kf = KalmanFilter([position[0], position[1], 0.0, 0.0], std_a_x, std_a_y, r_laser)

        self.age = 1
        self.lost_count = 0
        self.static_count = 0
        self.is_static = False
        self.state = "TENTATIVE"  # -> "ACTIVE" once confirmed

        self._smooth_vx = 0.0
        self._smooth_vy = 0.0

    def predict(self, dt):
        self.kf.predict(dt)

    def update(self, position):
        self.kf.update(position)
        self.lost_count = 0
        self.age += 1
        # Require consecutive matches before publishing: filters spurious 1-3 frame
        # DR-SPAAM blips from walls/glass rather than a real, persistent person.
        if self.state == "TENTATIVE" and self.age >= 5:
            self.state = "ACTIVE"

        vx, vy = self.velocity
        self._smooth_vx = self._VEL_ALPHA * vx + (1.0 - self._VEL_ALPHA) * self._smooth_vx
        self._smooth_vy = self._VEL_ALPHA * vy + (1.0 - self._VEL_ALPHA) * self._smooth_vy

    def check_static(self, speed_threshold, static_frames_required):
        if self.speed < speed_threshold:
            self.static_count += 1
        else:
            self.static_count = 0
            self.is_static = False

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
    def smoothed_velocity(self):
        return (self._smooth_vx, self._smooth_vy)


class MultiObjectTracker:
    """Associates per-frame detections into persistent tracks.

    ROS-free by design: fed with step(dt, detections_xy) and returns the current
    ACTIVE tracks, so it can be driven from a gRPC servicer thread just as easily
    as a topic callback.
    """

    def __init__(
        self,
        association_threshold=1.0,
        max_lost_frames=10,
        std_a_x=1.5,
        std_a_y=1.5,
        r_laser=0.2,
        static_speed_threshold=0.15,
        static_frames_required=15,
    ):
        self.association_threshold = association_threshold
        self.max_lost_frames = max_lost_frames
        self.std_a_x = std_a_x
        self.std_a_y = std_a_y
        self.r_laser = r_laser
        self.static_speed_threshold = static_speed_threshold
        self.static_frames_required = static_frames_required

        self.tracks = []
        self._next_track_id = 1

    def step(self, dt, detections_xy):
        """Advance the tracker by dt using this frame's (x, y) detections.

        Returns the list of currently ACTIVE (confirmed) Track objects, static
        ones included -- see the module docstring.
        """
        for track in self.tracks:
            track.predict(dt)

        associations, matched_tracks, matched_dets = self._associate(detections_xy)

        for t_idx, d_idx in associations:
            self.tracks[t_idx].update(detections_xy[d_idx])

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.lost_count += 1

        for d_idx, det in enumerate(detections_xy):
            if d_idx in matched_dets:
                continue
            self.tracks.append(Track(self._next_track_id, det, self.std_a_x, self.std_a_y, self.r_laser))
            self._next_track_id += 1

        active_tracks = []
        for track in self.tracks:
            # Still classified every frame so `is_static` stays current for any
            # consumer that wants it; it just no longer removes the track.
            track.check_static(self.static_speed_threshold, self.static_frames_required)
            if track.lost_count > self.max_lost_frames:
                continue
            active_tracks.append(track)
        self.tracks = active_tracks

        return [t for t in self.tracks if t.state == "ACTIVE"]

    def _associate(self, detections_xy):
        matched_tracks, matched_dets, associations = set(), set(), []
        if not self.tracks or not detections_xy:
            return associations, matched_tracks, matched_dets

        if HAS_SCIPY:
            cost = np.zeros((len(self.tracks), len(detections_xy)), dtype=np.float64)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections_xy):
                    cost[t_idx, d_idx] = np.hypot(track.position[0] - det[0], track.position[1] - det[1])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.association_threshold:
                    associations.append((r, c))
                    matched_tracks.add(r)
                    matched_dets.add(c)
        else:
            candidates = []
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections_xy):
                    dist = np.hypot(track.position[0] - det[0], track.position[1] - det[1])
                    if dist < self.association_threshold:
                        candidates.append((dist, t_idx, d_idx))
            candidates.sort(key=lambda c: c[0])
            for _, t_idx, d_idx in candidates:
                if t_idx not in matched_tracks and d_idx not in matched_dets:
                    associations.append((t_idx, d_idx))
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)

        return associations, matched_tracks, matched_dets
