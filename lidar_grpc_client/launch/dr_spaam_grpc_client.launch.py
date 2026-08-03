"""Full receiving pipeline: pull scans over gRPC -> DR-SPAAM detect -> KF track.

Brings up the gRPC client alongside the existing detector and tracker nodes, which run
completely unmodified — they just see a normal LaserScan topic. That topic is
/scan_remote, not /scan: if this laptop and the robot can also see each other over
native ROS2 DDS, the robot's own LD19 driver publishing directly to /scan would collide
with grpc_scan_client_node's republished copy on the same topic name (see the
scan_topic comment in lidar_grpc_client/config/params.yaml). Keeping the gRPC-relayed
stream on its own topic name avoids that regardless of DDS discovery configuration, so
dr_spaam_ros2_node's scan_topic parameter is overridden here to match.
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
        # base_link -> base_laser: static, from the physical LiDAR mount offset (same
        # values as dr_spaam_full_tracker.launch.py's base_link_to_base_laser_ld19).
        # Published locally here rather than relied on from the robot's own copy
        # (ld19.launch.py) because odom->base_link now arrives over the gRPC link
        # instead of native ROS2 DDS -- with DDS no longer crossing the network, the
        # robot's static transform wouldn't reach this machine either, and RViz/the
        # tracker need the full base_laser->base_link->odom chain locally to resolve.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_base_laser_ld19',
            arguments=['0', '0', '0.18', '0', '0', '0', 'base_link', 'base_laser'],
        ),
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
            # dr_spaam_config's scan_topic default ("/scan") is shared with the
            # single-machine dr_spaam_full_tracker.launch.py; override it here to match
            # grpc_scan_client_node's scan_topic ("/scan_remote") instead of changing
            # the shared default.
            parameters=[dr_spaam_config, {'scan_topic': '/scan_remote'}],
        ),
        Node(
            package='dr_spaam_ros2',
            executable='dr_spaam_tracker_node',
            name='dr_spaam_tracker_node',
            output='screen',
            parameters=[dr_spaam_config],
        ),
    ])
