source /opt/ros/jazzy/setup.bash
source /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/dr_spaam"
cd /media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/ros2_ws
source install/setup.bash
ros2 launch dr_spaam_ros2 dr_spaam_tracker_no_tf.launch.py