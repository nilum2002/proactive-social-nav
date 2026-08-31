#!/usr/bin/env bash
# Norfair tracker, pipelined variant.
#
#   node   : norfair_inf_server_pipelined_node
#   params : ros2_ws/src/benchmark/config/norfair_pipelined_params.yaml
#
# Tracking runs in the odometry frame (detections are projected by
# _to_tracking_frame before the tracker sees them), and track initiation and
# deletion are counter-based: c_init updates to initiate, c_del updates without
# a match to delete. Tune c_init against conf_thresh -- lower c_init reacts
# sooner but admits more false tracks.
#
# Any extra arguments are passed through to ros2 launch, e.g.
#   ./run_norfair_pipelined.sh publish_static_tf:=false
# when another benchmark launch already owns map->odom and base_link->laser.
# NOT `set -u`: /opt/ros/jazzy/setup.bash and venv/bin/activate both
# reference unbound variables (AMENT_TRACE_SETUP_FILES, PS1), so nounset
# aborts the script before anything runs.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=1

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

# dr_spaam from the repo, plus the venv packages (torch, norfair, ...) that the
# system python used by colcon cannot see on its own.
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/dr_spaam:$REPO_ROOT/venv/lib/python3.12/site-packages"

cd "$REPO_ROOT/ros2_ws"

colcon build --packages-select benchmark --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch benchmark norfair_pipelined.launch.py "$@"
