# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ROS 2 (Jazzy) project for proactive social navigation using 2D LiDAR-based pedestrian detection and tracking. The pipeline runs on a physical robot (Qbot) with an LDROBOT D500 LiDAR, detects people using the DR-SPAAM neural network, and tracks them with a Kalman filter to enable proactive navigation.

## Workspace Layout

- `dr_spaam/` — Pure-Python ML library (no ROS). Contains `dr_spaam/detector.py` (inference entrypoint), `dr_spaam/model/` (DROW3 and DR-SPAAM network architectures), and training/dataset utilities.
- `dr_spaam_ros2/` — Standalone ROS 2 package (can be built independently). Contains the detector and tracker nodes.
- `ros2_ws/` — Main colcon workspace. `src/` symlinks/contains: `dr_spaam_ros2`, `ldlidar_stl_ros2` (LiDAR driver), and `qbot_description` (robot URDF).
- `frog_dataset/` — Custom fine-tuned checkpoint (`dr_spaam_5_on_frog.pth`) trained on local data.
- `self_supervised_person_detection-*/` — Upstream pre-trained checkpoints from JRDB.
- `venv/` — Python 3.12 virtualenv for the ML library (PyTorch, NumPy, scipy, etc.).

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r req.txt
pip install -e dr_spaam/   # installs dr_spaam as editable package
```

## Building the ROS 2 Workspace

**Deactivate the venv before building** — CMake must use the system Python, not the venv Python:

```bash
deactivate
cd ros2_ws
rm -rf build/ install/ log/   # clean if switching environments
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

## Running the Full Pipeline

Each step requires its own terminal.

**Terminal 1 — LiDAR driver** (device defaults to `/dev/ttyUSB0`):
```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws && source install/setup.bash
ros2 launch ldlidar_stl_ros2 ld19.launch.py
```

**Terminal 2 — DR-SPAAM detector (+ optional tracker)**:
```bash
source /opt/ros/jazzy/setup.bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/path/to/proactive-social-nav/dr_spaam:/path/to/proactive-social-nav/venv/lib/python3.12/site-packages
cd ros2_ws && source install/setup.bash

# Detector only:
ros2 launch dr_spaam_ros2 dr_spaam_ros2.launch.py

# Detector + Kalman tracker:
ros2 launch dr_spaam_ros2 dr_spaam_full.launch.py

# Tracker only (expects detector already running):
ros2 launch dr_spaam_ros2 dr_spaam_tracker.launch.py
```

**Terminal 3 — RViz2 visualization**:
```bash
source /opt/ros/jazzy/setup.bash
rviz2
```
RViz2 setup: Fixed Frame → `base_laser`; Add → By topic: `/scan` (LaserScan), `/dr_spaam_ros2_node/rviz_marker` (Marker), `/dr_spaam_tracker_node/markers` (MarkerArray).

## Node Architecture

### `dr_spaam_ros2_node` (`dr_spaam_ros2/dr_spaam_ros2_node.py`)
- Subscribes: `/scan` (LaserScan)
- Publishes: `~/detections` (PoseArray), `~/rviz_marker` (Marker)
- Wraps `dr_spaam.detector.Detector`. On the first scan, calls `set_laser_fov()` to configure the model, then runs inference on every scan.
- Applies a configurable FOV crop (`scan_fov_deg`, default 270°) to remove the rear sector occluded by the robot body.

### `dr_spaam_tracker_node` (`dr_spaam_ros2/dr_spaam_tracker_node.py`)
- Subscribes: `/dr_spaam_ros2_node/detections` (PoseArray)
- Publishes: `~/tracks` (PoseArray — active confirmed tracks), `~/markers` (MarkerArray — cylinders, velocity arrows, predicted positions)
- Implements a 2D constant-velocity Kalman filter per track. Uses scipy Hungarian algorithm for data association (falls back to greedy nearest-neighbor if scipy unavailable).
- Track lifecycle: TENTATIVE (age < 5 consecutive matches) → ACTIVE. Tracks removed after `max_lost_frames` missed detections or if classified static (speed below `static_speed_threshold` for `static_frames_required` consecutive frames). Removed static positions are blacklisted for 5 s to suppress re-spawning.
- Adaptive process noise: computes the Normalized Innovation Squared (NIS) on each update; when NIS > 9.0 (99th percentile of chi²(2)), a maneuver is detected and Q is temporarily scaled up so velocity re-estimates quickly. Scale decays back to 1× at 0.85 per step.
- Prediction display uses an EMA-smoothed velocity (α = 0.35) stored separately from the KF state, so the RViz arrow is stable without affecting position tracking or data association.

## Key Configuration (`dr_spaam_ros2/config/params.yaml`)

**Detector node (`dr_spaam_ros2_node`):**

| Parameter | Value | Notes |
|---|---|---|
| `weight_file` | `frog_dataset/dr_spaam_5_on_frog.pth` | Custom fine-tuned checkpoint |
| `detector_model` | `DR-SPAAM` | `DROW3` or `DR-SPAAM` |
| `conf_thresh` | `0.7` | Detection confidence threshold |
| `scan_fov_deg` | `270.0` | FOV crop — remove rear 90° |
| `max_range` | `6.0` m | Cap detection distance; `-1.0` = use sensor hardware max (12 m) |

**Tracker node (`dr_spaam_tracker_node`):**

| Parameter | Value | Notes |
|---|---|---|
| `std_a_x` / `std_a_y` | `0.8` m/s² | Process noise — **must be > 0** or velocity estimate freezes |
| `r_laser` | `0.20` m | LiDAR measurement noise std dev |
| `association_threshold` | `1.0` m | Max matching distance for Hungarian assignment |
| `max_lost_frames` | `10` | Frames before track deletion |
| `prediction_horizon` | `1.0` s | Future position preview in RViz |
| `static_speed_threshold` | `0.1` m/s | Below this speed a track is considered static |
| `static_frames_required` | `8` | Consecutive static frames before track eviction |

## The `dr_spaam` ML Library

`dr_spaam/detector.py:Detector` is the only public interface needed at runtime. It accepts a raw range array and optional `scan_phi` angles, returns `(dets_xy, dets_cls, instance_mask)`. The two model choices are:
- **DROW3** (`dr_spaam/model/drow_net.py`) — single-scan detector.
- **DR-SPAAM** (`dr_spaam/model/dr_spaam.py`) — temporal, uses a sliding window of past scans for better accuracy.

Training scripts and dataset utilities live in `dr_spaam/bin/` and `dr_spaam/dr_spaam/dataset/`. Configs for training are in `dr_spaam/cfgs/`.

## Running Tests (ML Library)

Tests use `pytest` and require the venv active:
```bash
source venv/bin/activate
cd dr_spaam
pytest tests/test_detector.py     # basic detector smoke test
pytest tests/test_dataloader.py   # requires DROW dataset path configured
```

## Checkpoints

| File | Description |
|---|---|
| `frog_dataset/dr_spaam_5_on_frog.pth` | Fine-tuned on local "frog" robot data (active default) |
| `self_supervised_person_detection-*/ckpt_jrdb_ann_dr_spaam_e20.pth` | Upstream JRDB annotated checkpoint |
| `self_supervised_person_detection-*/ckpt_jrdb_pl_dr_spaam_e20.pth` | Upstream JRDB pseudo-label checkpoint |

Swap the `weight_file` in `config/params.yaml` to switch checkpoints without recompiling.

## Implementation Notes

See `CHANGES.md` for detailed reasoning behind every non-obvious design decision in the tracker — process noise tuning, NIS thresholds, ACTIVE gate value, EMA factor, and the static blacklist mechanism.
