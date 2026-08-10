source /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/bin/activat
cd ros2_ws/
export PYTHONPATH=$PYTHONPATH:/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/dr_spaam:/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/lib/python3.12/site-packages

colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

ros2 launch inf_server inf_server.launch.py