import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Sequential (non-pipelined) benchmark server."""

    package_name = 'benchmark'

    config_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'params.yaml'
    )

    publish_tf = LaunchConfiguration('publish_static_tf')

    return LaunchDescription([
        DeclareLaunchArgument(
            'publish_static_tf',
            default_value='true',
            description='Set false when another server launch is already publishing these',
        ),

        Node(
            package=package_name,
            executable='grpc_server_node',
            name='inf_server_node',
            output='screen',
            parameters=[config_path]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='benchmark_map_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen',
            condition=IfCondition(publish_tf),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='benchmark_laser_tf',
            arguments=['0.1', '0.0', '0.05', '0.0', '0.0', '0.0', 'base_link', 'laser'],
            output='screen',
            condition=IfCondition(publish_tf),
        ),
    ])
