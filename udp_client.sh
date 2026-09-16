
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash

export ROS_LOCALHOST_ONLY=1

cd "$REPO_ROOT/ros2_ws"
colcon build --packages-select inf_client_udp \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

PARAMS="$REPO_ROOT/ros2_ws/install/inf_client_udp/share/inf_client_udp/config/params.yaml"

exec ros2 run inf_client_udp udp_client_node \
    --ros-args --params-file "$PARAMS" "$@"
