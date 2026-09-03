#!/bin/bash
# Runs inf_client_wifi: republishes /scan and /odom onto subnet-visible DDS
# topics so inf_server can subscribe to them directly, with no custom transport.
#
# Run this ALONGSIDE a stack that owns the hardware (kobuki_stack.sh or
# nav2_stack.sh), the same way udp_client.sh is run.
#
#   ./wifi_client.sh                                  # params.yaml as-is
#   SERVER=192.168.0.42 ./wifi_client.sh              # different inf_server
#   ./wifi_client.sh -p output_reliability:=reliable  # measure DDS retransmit
#   ./wifi_client.sh -p scan_decimation:=4            # match a decimated UDP run
#
# No venv: this package is pure rclpy, no protobuf and no grpcio.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Host running inf_server. Discovery is unicast to this address, so it has to
# be right -- a wrong value fails as "no subscriber", not as a connection error.
SERVER="${SERVER:-192.168.0.100}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

source /opt/ros/jazzy/setup.bash

# THE critical line. Every other script here sets ROS_LOCALHOST_ONLY=1, and it
# is probably in ~/.bashrc too. It is honored over the newer discovery
# variables when enabled, so leaving it set confines this node to loopback and
# inf_server is never discovered -- with no error, just a topic nobody joins.
unset ROS_LOCALHOST_ONLY

# SUBNET rather than LOCALHOST: this participant has to reach both ways -- the
# driver stack on loopback and inf_server across the WiFi. STATIC_PEERS adds a
# direct unicast path to the server so discovery does not depend on multicast,
# which APs send at the lowest basic rate and never retry.
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export ROS_STATIC_PEERS="$SERVER"

cd "$REPO_ROOT/ros2_ws"
colcon build --packages-select inf_client_wifi \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

PARAMS="$REPO_ROOT/ros2_ws/install/inf_client_wifi/share/inf_client_wifi/config/params.yaml"

echo "inf_client_wifi -> domain $ROS_DOMAIN_ID, static peer $SERVER"

# ros2 run rather than the launch file, matching udp_client.sh: the launch file
# declares no arguments, so it cannot take the -p overrides above.
exec ros2 run inf_client_wifi dds_relay_node \
    --ros-args --params-file "$PARAMS" "$@"
