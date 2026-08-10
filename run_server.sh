#!/bin/bash
# Source ROS2 + venv (torch/dr_spaam/grpcio live only in the venv, not system
# python), then build and launch inf_server: DR-SPAAM + KF tracking over gRPC.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# inf_server and the robot must NOT discover each other's ROS graphs over the
# network -- all data crosses the machine boundary through the gRPC link on
# purpose, never native DDS. Restrict discovery to this machine only.
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=1

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/dr_spaam:$REPO_ROOT/venv/lib/python3.12/site-packages"

cd "$REPO_ROOT/ros2_ws"
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch inf_server inf_server.launch.py
