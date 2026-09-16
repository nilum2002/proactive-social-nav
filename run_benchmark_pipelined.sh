#!/usr/bin/env bash
# benchmark, gRPC pipelined variant. The instrumented sibling of
# run_server_pipelined.sh, and the pipelined counterpart to
# run_benchmark_seq.sh: detection (stage 1) and tracking (stage 2) run on
# separate threads with a bounded queue between them, and per-stage latency,
# CPU/RAM and GPU samples are written to
# sys_reports_server/inf_server_service_log_pipelined.csv.
#
#   node   : inf_server_pipelined_node (benchmark package)
#   params : ros2_ws/src/benchmark/config/pipelined_params.yaml
#   listens: grpc://0.0.0.0:50054  (50054, not 50053 -- run_benchmark_seq.sh
#                                   owns 50053, so both can run for an A/B)
#
# Unlike the UDP and WiFi pipelined nodes, a full queue here *blocks* stage 1
# rather than dropping: gRPC streams over TCP, so backpressure genuinely
# reaches the robot instead of being silently absorbed by a socket buffer.
#
#   ./run_benchmark_pipelined.sh
#   ./run_benchmark_pipelined.sh publish_static_tf:=false
#
# Pass publish_static_tf:=false when run_benchmark_seq.sh (or any other
# inf_server launch) is already up: both launches publish the same map->odom
# and base_link->laser static transforms, and two publishers fighting over one
# edge makes TF non-deterministic.
#
# NOT `set -u`: /opt/ros/jazzy/setup.bash and venv/bin/activate both reference
# unbound variables (AMENT_TRACE_SETUP_FILES, PS1), so nounset aborts before
# anything runs.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Scopes the ROS graph to this machine (RViz runs here). It does not affect the
# robot's feed: that arrives on a gRPC/TCP socket bound to 0.0.0.0, which DDS
# discovery settings have no say over.
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=1

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

# dr_spaam from the repo, plus the venv packages (torch, grpcio, ...) that the
# system python used by colcon cannot see on its own.
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/dr_spaam:$REPO_ROOT/venv/lib/python3.12/site-packages"

cd "$REPO_ROOT/ros2_ws"

colcon build --packages-select benchmark --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch benchmark benchmark_pipelined.launch.py "$@"
