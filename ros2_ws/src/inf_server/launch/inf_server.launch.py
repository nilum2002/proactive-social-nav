import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'inf_server'

    config_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
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
            name='inf_server_map_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='inf_server_laser_tf',
            arguments=['0.1', '0.0', '0.05', '0.0', '0.0', '0.0', 'base_link', 'laser'],
            output='screen',
        ),
    ])
