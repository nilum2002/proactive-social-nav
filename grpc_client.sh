source /opt/ros/jazzy/setup.bash
source /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/dr_spaam:$(python3 -c 'import site;print(site.getsitepackages()[0])')"
cd /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/ros2_ws
source install/setup.bash
ros2 launch lidar_grpc_client dr_spaam_grpc_client.launch.py