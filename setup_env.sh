#!/bin/bash
# 1. Create the venv (skipped if it already exists), activate it, install deps,
#    then source ROS2 and launch the LD19 driver.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# if [ ! -d venv ]; then
#     python3 -m venv venv
# fi
# source venv/bin/activate

# # pip install -r req.txt
# # cd dr_spaam

# # pip install -e dr_spaam




source /opt/ros/jazzy/setup.bash

cd "$REPO_ROOT/ros2_ws"
colcon build --symlink-install
source install/setup.bash

ros2 launch ldlidar_stl_ros2 ld19.launch.py
