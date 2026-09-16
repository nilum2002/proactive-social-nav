set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


SERVER="${SERVER:-192.168.0.100}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

source /opt/ros/jazzy/setup.bash


unset ROS_LOCALHOST_ONLY


export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export ROS_STATIC_PEERS="$SERVER"

cd "$REPO_ROOT/ros2_ws"
colcon build --packages-select inf_client_wifi \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

PARAMS="$REPO_ROOT/ros2_ws/install/inf_client_wifi/share/inf_client_wifi/config/params.yaml"

echo "inf_client_wifi -> domain $ROS_DOMAIN_ID, static peer $SERVER"


exec ros2 run inf_client_wifi dds_relay_node \
    --ros-args --params-file "$PARAMS" "$@"
