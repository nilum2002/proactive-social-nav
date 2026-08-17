import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Sequential (non-pipelined) benchmark server.

    Same node as inf_server's grpc_server_node, but the instrumented copy: it
    writes per-stage latency and CPU/GPU samples to the CSV named by
    service_log_file in config/params.yaml.
    """
    package_name = 'benchmark'

    config_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'params.yaml'
    )

    # The node name must stay in sync with the top-level key in params.yaml,
    # which is why it is still 'inf_server_node' and not 'benchmark_node'.
    #
    # This binds the same port (50053) as inf_server's own launch file, so the
    # two cannot run at once -- run this one *instead of* inf_server when you
    # want the CSV reports.
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
        # The server never gets the robot's real TF tree over gRPC (that only exists
        # on the robot's own ROS graph). It rebuilds an equivalent chain locally:
        # map->odom (identity, just so "odom" always shows up as a selectable Fixed
        # Frame even before the first /odom frame arrives) and odom->base_link is
        # broadcast dynamically by grpc_server_node itself from the odometry it
        # receives over gRPC. This leaves only base_link->laser, which is static.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='benchmark_map_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen',
            condition=IfCondition(publish_tf),
        ),
        # Laser mount offset relative to base_link -- a fixed calibration constant,
        # not live robot state, so it's safe to duplicate here. MUST be kept in sync
        # with the real value in kobuki_driver's teleop_stack_launch.py; if that
        # offset ever changes on the robot, update it here too.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='benchmark_laser_tf',
            arguments=['0.1', '0.0', '0.05', '0.0', '0.0', '0.0', 'base_link', 'laser'],
            output='screen',
            condition=IfCondition(publish_tf),
        ),
    ])
