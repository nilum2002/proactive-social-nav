import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/nilum/my-stuff/Research/Human_Robot_Interaction/2D_lidar_person_detection/dr_spaam_ros2/install/dr_spaam_ros2'
