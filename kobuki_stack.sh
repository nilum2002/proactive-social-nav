#!/bin/bash
# Source ROS2 + venv (teleop_stack_launch.py includes inf_client, which needs the
# grpcio only found in the venv), then build and launch the kobuki + lidar +
# inf_client stack for teleop operation.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

VENV_SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$PYTHONPATH:$VENV_SITE_PACKAGES"

cd "$REPO_ROOT/ros2_ws"
colcon build --packages-select kobuki_driver lidar_driver inf_client --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch kobuki_driver teleop_stack_launch.py
