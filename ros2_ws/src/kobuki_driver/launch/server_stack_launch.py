import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    kobuki_pkg_dir = get_package_share_directory('kobuki_driver')
    slam_launch = os.path.join(kobuki_pkg_dir, 'launch', 'slam_launch.py')
    nav2_coordinator_pkg_dir = get_package_share_directory('nav2_coordinator')
    nav2_coordinator_launch = os.path.join(nav2_coordinator_pkg_dir, 'launch', 'nav_coordinator_launch.py')
    rosbridge_pkg_dir = get_package_share_directory('rosbridge_server')
    rosbridge_launch = os.path.join(rosbridge_pkg_dir, 'launch', 'rosbridge_websocket_launch.xml')


    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),    
        
        # SLAM Toolbox
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments=[('use_sim_time', LaunchConfiguration('use_sim_time'))],
        ),

        # Nav2 Coordinator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_coordinator_launch),
            launch_arguments=[('use_sim_time', LaunchConfiguration('use_sim_time'))],
        ),
        
        # rosbridge websocket (for web UI)
        IncludeLaunchDescription(
            XMLLaunchDescriptionSource(rosbridge_launch),
        ),
    ])
