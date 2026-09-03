#!/usr/bin/env bash
# inf_server_udp, pipelined variant.
#
#   node   : inf_server_udp_pipelined_node
#   params : ros2_ws/src/inf_server_udp/config/pipelined_params.yaml
#   listens: udp://0.0.0.0:50054   (50054, not 50053 -- the gRPC inf_server can
#                                   keep running alongside for an A/B run)
#
# Stage 1 runs the detector, stage 2 runs the tracker and the publishers, with a
# bounded queue between them. There is no backpressure to give a UDP sender, so
# a scan arriving while the detector is busy replaces the one waiting and is
# counted as scans_dropped_busy; the server reports that, plus loss, reorder and
# jitter, back to the robot once a second as the client's only congestion signal.
#
# Extra arguments pass through to ros2 launch, e.g.
#   ./run_server_udp_pipelined.sh publish_static_tf:=false
# when another inf_server launch already owns map->odom and base_link->laser.
#
# NOT `set -u`: /opt/ros/jazzy/setup.bash and venv/bin/activate both reference
# unbound variables (AMENT_TRACE_SETUP_FILES, PS1), so nounset aborts before
# anything runs.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Scopes the ROS graph to this machine (RViz runs here). It does not affect the
# robot's UDP feed: that arrives on a raw socket bound to 0.0.0.0, which DDS
# discovery settings have no say over.
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=1

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

# dr_spaam from the repo, plus the venv packages (torch, protobuf, ...) that the
# system python used by colcon cannot see on its own.
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/dr_spaam:$REPO_ROOT/venv/lib/python3.12/site-packages"

cd "$REPO_ROOT/ros2_ws"

colcon build --packages-select inf_server_udp --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch inf_server_udp inf_server_udp_pipelined.launch.py "$@"
