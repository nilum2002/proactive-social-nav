#!/usr/bin/env bash
# benchmark, UDP pipelined variant. The instrumented sibling of
# run_server_udp_pipelined.sh: detection (stage 1) and tracking (stage 2) run
# on separate threads with a bounded queue, and per-scan latency (including
# the queueing delay), loss/jitter and CPU/GPU samples are written to
# sys_reports_server/inf_server_service_log_udp_pipelined.csv.
#
#   node   : inf_server_udp_pipelined_node (benchmark package)
#   params : ros2_ws/src/benchmark/config/udp_pipelined_params.yaml
#   listens: udp://0.0.0.0:50054   (as run_benchmark_udp_seq.sh -- only one
#                                   UDP server can hold the port at a time)
#
#   ./run_benchmark_udp_pipelined.sh
#   ./run_benchmark_udp_pipelined.sh publish_static_tf:=false
#
# NOT `set -u`: see run_benchmark_udp_seq.sh.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=1

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/dr_spaam:$REPO_ROOT/venv/lib/python3.12/site-packages"

cd "$REPO_ROOT/ros2_ws"

colcon build --packages-select benchmark --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch benchmark benchmark_udp_pipelined.launch.py "$@"
