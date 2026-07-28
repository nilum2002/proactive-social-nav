# DR-SPAAM Tracker & Detector — Changes

Documents every change made to `dr_spaam_tracker_node.py`, `dr_spaam_ros2_node.py`, and `config/params.yaml`, with the reasoning behind each decision.

---

## Issue 1 — False Positives from DR-SPAAM

### 1.1 Stricter TENTATIVE → ACTIVE gate

**Problem:** A new track starts in TENTATIVE state and becomes ACTIVE after N consecutive matched detections. With N=3 frames (~0.3 s at 10 Hz), false positives from walls, glass panels, and furniture — which typically appear for 1–3 scan frames then vanish — were reaching ACTIVE state and being published on the `~/tracks` topic.

**Method:** Raised the promotion threshold from 3 to 5 consecutive matched frames. A real pedestrian will be detected consistently across those 5 frames. A reflective surface or static object typically produces 1–3 detections before the LiDAR ray angle shifts past it.

---

### 1.2 Faster static object eviction

**Problem:** The static object filter removes tracks whose estimated speed stays below `static_speed_threshold` for `static_frames_required` consecutive frames. With the threshold at 15 frames (1.5 s), a false positive could become ACTIVE at frame 5 and remain ACTIVE until frame 15 — publishing a ghost track for ~1 second.

**Method:** Reduced `static_frames_required` from 15 to 8 frames. This cuts the false-positive publish window to ~0.3 s after it becomes ACTIVE. Evicted static positions are added to a 5-second blacklist so the same location cannot immediately re-spawn a new track.

---

## Issue 2 — Direction Change Lag

### 2.1 Fixed frozen velocity (Q = 0 bug)

**Problem:** `std_a_x` and `std_a_y` were set to `0.0` in `params.yaml`. These values feed directly into the process noise matrix Q. When Q = 0, the Kalman gain K becomes effectively zero, meaning the velocity components `[vx, vy]` of the state vector stop updating after initial convergence. The filter would settle on an early velocity estimate and refuse to change it — causing severe lag whenever a person turned or changed speed. The code itself had a comment warning about this exact issue.

**Method:** Set `std_a_x` and `std_a_y` to `0.8 m/s²`. This value represents the expected unmodelled acceleration budget a pedestrian can sustain during normal indoor walking. It is high enough to allow velocity adaptation but low enough not to inject excessive noise into the position estimate.

---

### 2.2 Adaptive Q via Normalized Innovation Squared (NIS)

**Problem:** Even with a non-zero `std_a`, a Constant Velocity (CV) model inherently lags sharp direction changes by several frames. The model assumes constant velocity, so it takes multiple measurement updates before the filter accepts that the person has genuinely changed direction.

**Method:** Maneuver detection using the Normalized Innovation Squared (NIS). After each LiDAR update, the filter computes `NIS = yᵀ S⁻¹ y`, where `y` is the innovation (measurement minus prediction) and `S` is the innovation covariance. Under a correct model, NIS follows a chi-squared distribution with 2 degrees of freedom (mean ≈ 2). When NIS exceeds 9.0 (the 99th percentile of chi²(2)), the model is statistically inconsistent with the measurement — a maneuver has been detected.

On detection, the process noise Q is temporarily scaled up (capped at 15×). This allows the velocity state to update much faster than normal, rapidly re-estimating the new direction. The scale decays back to 1× over approximately 1.5 seconds as the filter re-converges on the new trajectory.

---

## Issue 3 — Runtime Crash on Every Callback

### 3.1 Numpy scalar conversion fix

**Problem:** After the NIS computation was added, the tracker crashed on every callback with the error: `only 0-dimensional arrays can be converted to Python scalars`. The NIS expression `yᵀ S⁻¹ y` where `y` is shape `(2, 1)` produces a `(1, 1)` shaped numpy array. Python's `float()` only accepts 0-dimensional scalars and raises a `TypeError` on `(1, 1)` arrays.

**Method:** Replaced `float(expression)` with `expression.item()`. The `.item()` method on a numpy array extracts a Python scalar from any single-element array regardless of its shape — `(1, 1)`, `(1,)`, or `()` all work correctly.

---

## Feature — Configurable Max LiDAR Range

### 4.1 max_range parameter

**Motivation:** The LDROBOT LD19 has a hardware range of 12 m. DR-SPAAM's reliable detection range is approximately 6–8 m; beyond that, a person's angular footprint in the scan becomes too small for reliable detection and false positive rates from distant walls and surfaces increase. There was previously no way to cap this without modifying code.

**Method:** Added a `max_range` parameter to the detector node. Any range reading beyond this value is replaced with the background sentinel value (29.99 m) before the scan is passed to DR-SPAAM, so the model treats those regions as empty space. Setting `max_range: -1.0` disables the cap and uses the sensor's full hardware range. Default is set to `6.0 m`.

---

## Improvement — Prediction Accuracy

### 5.1 Reduced prediction horizon

**Problem:** At 1.5 s, the linear CV extrapolation drifts noticeably from reality. Humans decelerate, curve, and deviate from constant velocity over that window — the further ahead you predict with a CV model, the larger the error compounds.

**Method:** Reduced `prediction_horizon` from 1.5 s to 1.0 s. The prediction is displayed in RViz as a ghost sphere and a path line. A shorter horizon gives a more accurate near-term prediction at the cost of slightly less "look-ahead" distance.

---

### 5.2 Velocity EMA for prediction smoothing

**Problem:** The raw Kalman filter velocity estimate `[vx, vy]` varies slightly frame-to-frame even during smooth steady walking, causing the RViz prediction arrow to wobble.

**Method:** Applied an Exponential Moving Average (EMA) to the velocity used for drawing the prediction markers. The formula is:

```
smooth_v(t) = α × kf_v(t) + (1 − α) × smooth_v(t−1)
```

With `α = 0.35`, recent frames are weighted more than old ones but rapid frame-to-frame jitter is suppressed. This smoothed velocity is stored separately and is used only for the RViz visualisation (arrow, predicted sphere, prediction path). The KF state itself, data association, static detection, and the `~/tracks` topic are all unaffected.

---

## Parameter Summary

| Parameter | Before | After | Location |
|---|---|---|---|
| `std_a_x` | `0.0` | `0.8` | `params.yaml` |
| `std_a_y` | `0.0` | `0.8` | `params.yaml` |
| `static_frames_required` | `15` | `8` | `params.yaml` |
| `prediction_horizon` | `1.5` | `1.0` | `params.yaml` |
| `max_range` | *(new)* | `6.0` | `params.yaml` |
| ACTIVE gate (frames) | `3` | `5` | code |
| NIS maneuver threshold | *(new)* | `9.0` | code |
| Velocity EMA α | *(new)* | `0.35` | code |
