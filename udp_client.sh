#!/bin/bash
# Runs inf_client_udp on its own: subscribes to /scan and /odom from whichever
# robot stack is already up and forwards them to inf_server_udp as datagrams.
#
# Run this ALONGSIDE a stack that owns the hardware (kobuki_stack.sh or
# nav2_stack.sh) -- never on its own, or there is no /scan to forward, and never
# as a second copy of a stack that already starts a client.
#
#   ./udp_client.sh                                    # config/params.yaml as-is
#   ./udp_client.sh -p server_address:=192.168.0.42:50054
#   ./udp_client.sh -p adaptive_rate:=false -p dscp:=34
#
# Unlike kobuki_stack.sh this does NOT source venv/. inf_client needs grpcio,
# which only exists in the venv; inf_client_udp needs protobuf + numpy, which
# only exist in the SYSTEM python (the venv has no protobuf at all). Activating
# the venv here would break the import.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash

# The robot stacks all run with this set. The node has to match to discover
# /scan and /odom over DDS -- it does not affect the raw UDP socket, which
# still reaches inf_server_udp on another machine.
export ROS_LOCALHOST_ONLY=1

cd "$REPO_ROOT/ros2_ws"
colcon build --packages-select inf_client_udp \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

PARAMS="$REPO_ROOT/ros2_ws/install/inf_client_udp/share/inf_client_udp/config/params.yaml"

# ros2 run rather than the launch file: inf_client_udp.launch.py declares no
# launch arguments, so it cannot take the -p overrides above. The launch file
# adds nothing else -- same executable, same node name, same params file.
exec ros2 run inf_client_udp udp_client_node \
    --ros-args --params-file "$PARAMS" "$@"
