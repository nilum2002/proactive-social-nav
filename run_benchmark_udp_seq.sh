#!/usr/bin/env bash
# benchmark, UDP seq variant. The instrumented sibling of run_server_udp.sh:
# same UDP transport (inf_client_udp -> udp://0.0.0.0:50054), same detect-then-
# track arrangement, but writes per-scan latency, loss/jitter and CPU/GPU
# samples to sys_reports_server/inf_server_service_log_udp_seq.csv.
#
#   node   : inf_server_udp_node (benchmark package)
#   params : ros2_ws/src/benchmark/config/udp_params.yaml
#   listens: udp://0.0.0.0:50054   (what inf_client_udp sends to -- the same
#                                   port inf_server_udp uses, so run this
#                                   one INSTEAD of that one, not alongside)
#
#   ./run_benchmark_udp_seq.sh
#   ./run_benchmark_udp_seq.sh publish_static_tf:=false   # another launch owns the TFs
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

colcon build --packages-select benchmark --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch benchmark benchmark_udp_server.launch.py "$@"
