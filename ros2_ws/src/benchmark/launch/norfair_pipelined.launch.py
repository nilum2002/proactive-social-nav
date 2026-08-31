import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'benchmark'

    config_path = os.path.join(
        get_package_share_directory(package_name), 'config', 'norfair_pipelined_params.yaml')

    # Same caveat as the other benchmark launch files: only one launch may own
    # the static TFs, or two publishers fight over map->odom and base_link->laser.
    publish_tf = LaunchConfiguration('publish_static_tf')

    return LaunchDescription([
        DeclareLaunchArgument(
            'publish_static_tf',
            default_value='true',
            description='Set false when another benchmark launch already publishes these',
        ),

        Node(
            package=package_name,
            executable='norfair_pipelined_node',
            name='norfair_inf_server_pipelined_node',
            output='screen',
            parameters=[config_path]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='norfair_pipelined_map_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen',
            condition=IfCondition(publish_tf),
        ),
        # Laser mount offset relative to base_link -- keep in sync with
        # kobuki_driver's teleop_stack_launch.py if it ever changes on the robot.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='norfair_pipelined_laser_tf',
            arguments=['0.1', '0.0', '0.05', '0.0', '0.0', '0.0', 'base_link', 'laser'],
            output='screen',
            condition=IfCondition(publish_tf),
        ),
    ])
