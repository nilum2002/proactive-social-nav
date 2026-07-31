import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'lidar_grpc_client'

    config_path = os.path.join(
        get_package_share_directory(package_name), 'config', 'params.yaml'
    )

    return LaunchDescription([
        Node(
            package=package_name,
            executable='grpc_scan_client_node',
            name='grpc_scan_client_node',
            output='screen',
            parameters=[config_path],
        )
    ])
