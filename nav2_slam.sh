#!/bin/bash
# Full Nav2 stack with live SLAM (slam_toolbox) — no pre-saved map needed.
# Kobuki base + LD19 lidar + twist_mux + Nav2 (planner/controller/costmaps) +
# slam_toolbox building the map on the fly.
set -e

cd /home/nilum/proactive-social-nav
source /opt/ros/jazzy/setup.bash
cd ros2_ws

colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.bash
export ROS_LOCALHOST_ONLY=1

sudo chmod 666 /dev/ttyUSB* 2>/dev/null || true

# ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose   "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"   --feedback
if [ -e /dev/ldlidar ]; then
  LIDAR_PORT="/dev/ldlidar"
else
  echo "WARNING: /dev/ldlidar missing — run 'sudo ./setup_udev.sh'. Falling back to CP210 scan."
  LIDAR_PORT=""
  for dev in /dev/ttyUSB*; do
    if [ -e "$dev" ] && udevadm info --query=property --name="$dev" 2>/dev/null | grep -q "CP210"; then
      LIDAR_PORT="$dev"
      break
    fi
  done
fi
PORT="${1:-${LIDAR_PORT:-/dev/ttyUSB0}}"

echo "Launching Nav2 + live SLAM (lidar on $PORT -> $(readlink -f "$PORT" 2>/dev/null || echo "$PORT")) ..."
exec ros2 launch botzilla_nav nav2_bringup.launch.py slam:=True lidar_port:="$PORT"
