source /opt/ros/jazzy/setup.bash
source /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/dr_spaam:$(python3 -c 'import site;print(site.getsitepackages()[0])')"
cd /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/ros2_ws
# -DPython3_EXECUTABLE is required for the ament_cmake message package. This script
# activates the venv, and there is also a uv-managed python3.9 in ~/.local/bin; CMake
# picks one of those and then fails to import ROS's own 'em' (empy), which only exists
# for the system python3.12 that Jazzy was built against.
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch lidar_grpc_client dr_spaam_grpc_client.launch.py