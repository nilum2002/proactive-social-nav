#!/usr/bin/env bash
# benchmark, WiFi/DDS pipelined variant. The instrumented sibling of
# run_server_wifi_pipelined.sh: detection (stage 1) and tracking (stage 2) run
# on separate threads with a bounded queue, and per-scan latency (including
# the queueing delay), stamp-gap loss estimate, jitter and CPU/GPU samples are
# written to sys_reports_server/inf_server_service_log_wifi_pipelined.csv.
#
#   node   : inf_server_wifi_pipelined_node (benchmark package)
#   params : ros2_ws/src/benchmark/config/wifi_pipelined_params.yaml
#   listens: /wifi/scan + /wifi/odom  (DDS, BEST_EFFORT, keep-last 1)
#
#   ./run_benchmark_wifi_pipelined.sh                            # robot at $ROBOT below
#   ROBOT=192.168.0.55 ./run_benchmark_wifi_pipelined.sh
#   ./run_benchmark_wifi_pipelined.sh publish_static_tf:=false   # another launch owns the TFs
#
# NOT `set -u`: see run_benchmark_wifi_seq.sh.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOT="${ROBOT:-192.168.0.200}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

# See run_benchmark_wifi_seq.sh: this node's feed IS DDS, so
# ROS_LOCALHOST_ONLY must stay unset or the robot is never discovered.
unset ROS_LOCALHOST_ONLY

export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export ROS_STATIC_PEERS="$ROBOT"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/dr_spaam:$REPO_ROOT/venv/lib/python3.12/site-packages"

cd "$REPO_ROOT/ros2_ws"

colcon build --packages-select benchmark --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

echo "benchmark/wifi (pipelined) -> domain $ROS_DOMAIN_ID, static peer $ROBOT"

ros2 launch benchmark benchmark_wifi_pipelined.launch.py "$@"
