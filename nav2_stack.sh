#!/bin/bash
# Phase 2: navigate on the saved map.
#
# Sources ROS 2 + venv (nav_coordinator pulls in the same deps as the teleop
# stack), builds the nav packages, then brings up kobuki + lidar + map_server +
# amcl + nav2 + the navigate_to_coordinate service.
#
#   ./nav2_stack.sh                    # localize against maps/lab.yaml
#   ./nav2_stack.sh map:=maps/foo.yaml # a different map
#   ./nav2_stack.sh slam:=true         # phase 1 instead: build a map live
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP="${MAP:-$REPO_ROOT/maps/lab.yaml}"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

VENV_SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$PYTHONPATH:$VENV_SITE_PACKAGES"

export ROS_LOCALHOST_ONLY=1

cd "$REPO_ROOT/ros2_ws"
colcon build --packages-select kobuki_driver lidar_driver \
    nav_coordinator_interfaces nav2_coordinator \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

# Only inject the default map when the caller did not pass one.
for arg in "$@"; do
    case "$arg" in map:=*) MAP=""; break ;; esac
done

if [ -n "$MAP" ]; then
    ros2 launch nav2_coordinator nav2_bringup_launch.py map:="$MAP" "$@"
else
    ros2 launch nav2_coordinator nav2_bringup_launch.py "$@"
fi
