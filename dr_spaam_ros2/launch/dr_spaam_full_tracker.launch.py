#!/usr/bin/env python3
"""Full pedestrian tracking stack: LD19 LiDAR + DR-SPAAM detector + KF tracker."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('dr_spaam_ros2'),
        'config',
        'params.yaml'
    )

    # ── 1. LD19 LiDAR publisher ───────────────────────────────────────────────
    lidar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='LD19',
        output='screen',
        parameters=[
            {'product_name': 'LDLiDAR_LD19'},
            {'topic_name': 'scan'},
            {'frame_id': 'base_laser'},
            {'port_name': '/dev/ttyUSB1'},
            {'port_baudrate': 230400},
            {'laser_scan_dir': True},
            {'enable_angle_crop_func': False},
            {'angle_crop_min': 135.0},
            {'angle_crop_max': 225.0},
        ]
    )

    # ── 2. base_link → base_laser static TF ──────────────────────────────────
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_base_laser_ld19',
        arguments=['0', '0', '0.18', '0', '0', '0', 'base_link', 'base_laser']
    )

    # ── 3. DR-SPAAM pedestrian detector ──────────────────────────────────────
    detector_node = Node(
        package='dr_spaam_ros2',
        executable='dr_spaam_ros2_node',
        name='dr_spaam_ros2_node',
        output='screen',
        parameters=[params_file]
    )

    # ── 4. Kalman Filter multi-object tracker ─────────────────────────────────
    tracker_node = Node(
        package='dr_spaam_ros2',
        executable='dr_spaam_tracker_node',
        name='dr_spaam_tracker_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        lidar_node,
        tf_node,
        detector_node,
        tracker_node,
    ])
