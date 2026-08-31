"""Norfair-backed tracker exposing the same interface as MultiObjectTracker.

Drop-in for `kalman_tracker.MultiObjectTracker`: same `step(dt, detections_xy)`
call, same duck-typed track objects (`.id`, `.position`, `.velocity`, `.speed`),
so the gRPC servicers and the ROS publishers work against either unchanged.

Why Norfair maps cleanly onto the count-based init/delete policy
---------------------------------------------------------------
Norfair keeps exactly the counter the policy describes. Each update runs
`tracker_step()`, which does `hit_counter -= 1` for every object, and a matched
object then gets `hit_counter += 2 * period` in `hit()` -- a net +1 on a match
and -1 on a miss, capped at `hit_counter_max`.

    C_init  ->  initialization_delay
                A new detection creates a candidate immediately, but
                `is_initializing` stays True while `hit_counter <=
                initialization_delay`, and Tracker.update() does not return
                initialising objects. So a track becomes "properly initiated"
                only once the counter has climbed past C_init.

    C_del   ->  hit_counter_max
                The counter saturates at this value, so an established track
                survives exactly this many consecutive misses before
                `hit_counter` goes negative and the object is dropped.

Track initiation latency is therefore C_init scans (~C_init / scan_rate
seconds), which is the quantity to minimise for obstacle avoidance -- traded
against false positives, together with the detector confidence threshold.

Frame of reference
------------------
This class tracks in whatever frame it is handed. The node transforms
detections into the odometry frame before calling `step()` (see
`InfServerNode._to_tracking_frame`), which is what makes the velocity estimates
meaningful while the robot is moving.
"""
import numpy as np

from norfair import Detection, Tracker as _NorfairTracker


class _Track:
    """Adapter over norfair's TrackedObject with MultiObjectTracker's surface."""

    __slots__ = ("id", "position", "velocity", "speed", "hit_counter", "age", "is_static")

    def __init__(self, obj, dt):
        self.id = int(obj.id)
        est = np.asarray(obj.estimate, dtype=float).reshape(-1)
        self.position = (float(est[0]), float(est[1]))
        # norfair's filter advances once per update, so estimate_velocity is in
        # units-per-update; dividing by dt puts it in m/s to match the KF
        # tracker, whose velocity the RViz arrows and `speed` gate assume.
        vel = np.asarray(obj.estimate_velocity, dtype=float).reshape(-1)
        scale = 1.0 / dt if dt > 1e-6 else 0.0
        self.velocity = (float(vel[0]) * scale, float(vel[1]) * scale)
        self.speed = float(np.hypot(*self.velocity))
        self.hit_counter = int(obj.hit_counter)
        self.age = int(obj.age)
        # Kept so consumers written against the KF tracker keep working; norfair
        # has no static-object notion, and nothing here suppresses static people.
        self.is_static = False


class NorfairMultiObjectTracker:
    """Count-based person tracker built on norfair.

    Parameters
    ----------
    association_threshold : float
        Gating distance in metres (norfair's `distance_threshold`).
    c_init : int
        Updates a candidate must survive before its track is properly
        initiated. Lower = faster initiation = shorter reaction time, at the
        cost of more false tracks.
    c_del : int
        Updates without a match an established track tolerates before deletion.
    nominal_dt : float
        Expected scan period. Used only to convert an unusually long gap into
        norfair's integer `period`, so a dropped scan ages tracks correctly
        instead of being treated as a normal step.
    """

    def __init__(self, association_threshold=1.0, c_init=3, c_del=10,
                 nominal_dt=0.1, **_ignored):
        self.association_threshold = float(association_threshold)
        self.c_init = int(c_init)
        self.c_del = int(c_del)
        self.nominal_dt = float(nominal_dt)
        # norfair requires 0 <= initialization_delay < hit_counter_max. Checked
        # here with a message that names the ROS parameters, because otherwise
        # this surfaces as a bare ValueError on a gRPC worker thread the first
        # time a robot connects, long after the node looked healthy.
        if not 0 <= self.c_init < self.c_del:
            raise ValueError(
                f"c_init ({self.c_init}) must satisfy 0 <= c_init < c_del "
                f"({self.c_del}): a track cannot need more updates to initiate "
                f"than it survives without a match.")
        self._tracker = _NorfairTracker(
            distance_function="euclidean",
            distance_threshold=self.association_threshold,
            initialization_delay=self.c_init,
            hit_counter_max=self.c_del,
        )

    def step(self, dt, detections_xy):
        """Advance one scan. Returns the properly-initiated tracks only."""
        dets = [Detection(points=np.array([[float(x), float(y)]], dtype=float))
                for x, y in detections_xy]
        # period counts how many nominal scan intervals this update spans, so a
        # gap ages the counters by the right amount rather than by one.
        period = 1
        if self.nominal_dt > 1e-6:
            period = max(1, int(round(dt / self.nominal_dt)))
        # norfair only returns objects past initialization_delay, which is
        # exactly the "properly initiated" set.
        objs = self._tracker.update(detections=dets, period=period)
        return [_Track(o, dt) for o in objs]

    @property
    def tracks(self):
        return list(self._tracker.tracked_objects)
