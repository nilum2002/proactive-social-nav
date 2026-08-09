import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    kobuki_pkg_dir = get_package_share_directory('kobuki_driver')
    slam_config = os.path.join(kobuki_pkg_dir, 'config', 'slam_async_config.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),
        
        # SLAM Toolbox node in async mode
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_config, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[
                ('/scan', '/scan'),
                ('/odom', '/odometry/filtered'),
            ],
        ),
    ])
