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
        # inf_server never gets the robot's real TF tree over gRPC (that only exists
        # on the robot's own ROS graph). It rebuilds an equivalent chain locally:
        # map->odom (identity, just so "odom" always shows up as a selectable Fixed
        # Frame even before the first /odom frame arrives) and odom->base_link is
        # broadcast dynamically by grpc_server_node itself from the odometry it
        # receives over gRPC. This leaves only base_link->laser, which is static.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='inf_server_map_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen',
        ),
        # Laser mount offset relative to base_link -- a fixed calibration constant,
        # not live robot state, so it's safe to duplicate here. MUST be kept in sync
        # with the real value in kobuki_driver's teleop_stack_launch.py; if that
        # offset ever changes on the robot, update it here too.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='inf_server_laser_tf',
            arguments=['0.1', '0.0', '0.05', '0.0', '0.0', '0.0', 'base_link', 'laser'],
            output='screen',
        ),
    ])
