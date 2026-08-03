#!/bin/bash
# Source ROS2 + venv (grpcio lives only in the venv, not system python), then source
# the workspace and launch the LaserScan -> gRPC bridge. Consumers dial in and pull.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/venv/bin/activate"

# Keep this process's DDS participant off the network entirely -- odometry now
# reaches the laptop over the gRPC link (OdometryService/StreamOdometry), so native
# ROS2 discovery crossing the network would just create a second, uncoordinated
# publisher of the same odom->base_link transform on the laptop's TF tree. Must be
# set on BOTH machines, and in every terminal that touches ROS2 there (e.g. a
# standalone rviz2), not just the one running this script.
export ROS_LOCALHOST_ONLY=1

VENV_SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$PYTHONPATH:$VENV_SITE_PACKAGES"

cd "$REPO_ROOT/ros2_ws"
source install/setup.bash

ros2 launch lidar_grpc_bridge lidar_grpc_bridge.launch.py
