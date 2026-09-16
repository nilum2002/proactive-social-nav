#!/usr/bin/env bash
# benchmark, WiFi/DDS seq variant. The instrumented sibling of
# run_server_wifi.sh: same plain-DDS transport (/wifi/scan + /wifi/odom), same
# detect-then-track arrangement, but writes per-scan latency, stamp-gap loss
# estimate, jitter and CPU/GPU samples to
# sys_reports_server/inf_server_service_log_wifi_seq.csv.
#
#   node   : inf_server_wifi_node (benchmark package)
#   params : ros2_ws/src/benchmark/config/wifi_params.yaml
#   listens: /wifi/scan + /wifi/odom  (DDS, BEST_EFFORT, keep-last 1)
#
#   ./run_benchmark_wifi_seq.sh                            # robot at $ROBOT below
#   ROBOT=192.168.0.55 ./run_benchmark_wifi_seq.sh
#   ./run_benchmark_wifi_seq.sh publish_static_tf:=false   # another launch owns the TFs
#
# NOT `set -u`: /opt/ros/jazzy/setup.bash and venv/bin/activate both reference
# unbound variables (AMENT_TRACE_SETUP_FILES, PS1), so nounset aborts before
# anything runs.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The robot running inf_client_wifi. Discovery is unicast to this address, so it
# has to be right -- a wrong value fails as "no publisher discovered", never as
# a connection error.
ROBOT="${ROBOT:-192.168.0.200}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

# THE critical line, and the one difference from run_benchmark_udp*.sh here.
# The gRPC and UDP benchmark nodes set ROS_LOCALHOST_ONLY=1 because their feed
# arrives on a socket DDS has no say over; this node's feed IS DDS. Leaving it
# set -- and it is probably in ~/.bashrc too -- confines this node to loopback,
# the robot is never discovered, and there is no error: just a topic nobody
# publishes to.
unset ROS_LOCALHOST_ONLY

# SUBNET so the participant reaches across the WiFi. STATIC_PEERS adds a direct
# unicast path to the robot so discovery does not depend on multicast, which APs
# send at the lowest basic rate and never retry.
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export ROS_STATIC_PEERS="$ROBOT"

# dr_spaam from the repo, plus the venv packages (torch, ...) that the system
# python used by colcon cannot see on its own.
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/dr_spaam:$REPO_ROOT/venv/lib/python3.12/site-packages"

cd "$REPO_ROOT/ros2_ws"

colcon build --packages-select benchmark --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

echo "benchmark/wifi (seq) -> domain $ROS_DOMAIN_ID, static peer $ROBOT"

ros2 launch benchmark benchmark_wifi_server.launch.py "$@"
