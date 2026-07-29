#!/bin/bash
# Source ROS2 + venv, wire up PYTHONPATH for the local dr_spaam ML module and its
# venv deps, then source the workspace and launch the DR-SPAAM detector node.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

VENV_SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/dr_spaam:$VENV_SITE_PACKAGES"

cd "$REPO_ROOT/ros2_ws"
source install/setup.bash

ros2 launch dr_spaam_ros2 dr_spaam_full_tracker.launch.py
