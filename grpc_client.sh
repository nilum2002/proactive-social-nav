#!/bin/bash
# Source ROS2 + venv (grpcio lives only in the venv, not system python), then build
# and launch inf_client: forwards /scan and /odom to inf_server over gRPC.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

VENV_SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$PYTHONPATH:$VENV_SITE_PACKAGES"

cd "$REPO_ROOT/ros2_ws"
colcon build --packages-select inf_client --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch inf_client inf_client.launch.py
