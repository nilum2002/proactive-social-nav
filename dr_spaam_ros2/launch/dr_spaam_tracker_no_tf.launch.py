#!/usr/bin/env python3
"""DR-SPAAM detector, KF tracker no TF , for isolating KF tuning from TF/odom pipeline
issues. Assumes a LaserScan is already available on the configured scan_topic — from a
local LiDAR driver, or from lidar_grpc_client if run alongside it — this launch file
does not start a LiDAR driver or the gRPC client itself.

dr_spaam_tracker_no_tf_node starts on clean, untuned defaults (not the heavily-tuned
dr_spaam_ros2/config/params.yaml tracker block) — see its docstring for parameter names
if you want to override any via an additional --params-file or -p.

Only correct only while the robot is stationary — see dr_spaam_tracker_no_tf_node.py.

For Rviz visualization, add a MarkerArray display with topic /dr_spaam_tracker_no_tf_node/markers
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    dr_spaam_config = os.path.join(
        get_package_share_directory('dr_spaam_ros2'), 'config', 'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='dr_spaam_ros2',
            executable='dr_spaam_ros2_node',
            name='dr_spaam_ros2_node',
            output='screen',
            parameters=[dr_spaam_config],
        ),
        Node(
            package='dr_spaam_ros2',
            executable='dr_spaam_tracker_no_tf_node',
            name='dr_spaam_tracker_no_tf_node',
            output='screen',
        ),
    ])
