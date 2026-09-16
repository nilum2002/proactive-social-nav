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

for arg in "$@"; do
    case "$arg" in map:=*) MAP=""; break ;; esac
done

if [ -n "$MAP" ]; then
    ros2 launch nav2_coordinator nav2_bringup_launch.py map:="$MAP" "$@"
else
    ros2 launch nav2_coordinator nav2_bringup_launch.py "$@"
fi
