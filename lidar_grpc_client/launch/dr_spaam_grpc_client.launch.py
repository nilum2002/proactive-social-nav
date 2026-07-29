"""Full receiving pipeline: pull scans over gRPC -> DR-SPAAM detect -> KF track.

Brings up the gRPC client alongside the existing detector and tracker nodes, which run
completely unmodified — they just see a normal /scan topic.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    client_config = os.path.join(
        get_package_share_directory('lidar_grpc_client'), 'config', 'params.yaml'
    )
    dr_spaam_config = os.path.join(
        get_package_share_directory('dr_spaam_ros2'), 'config', 'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='lidar_grpc_client',
            executable='grpc_scan_client_node',
            name='grpc_scan_client_node',
            output='screen',
            parameters=[client_config],
        ),
        Node(
            package='dr_spaam_ros2',
            executable='dr_spaam_ros2_node',
            name='dr_spaam_ros2_node',
            output='screen',
            parameters=[dr_spaam_config],
        ),
        Node(
            package='dr_spaam_ros2',
            executable='dr_spaam_tracker_node',
            name='dr_spaam_tracker_node',
            output='screen',
            parameters=[dr_spaam_config],
        ),
    ])
