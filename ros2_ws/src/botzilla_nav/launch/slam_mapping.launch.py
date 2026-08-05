"""
SLAM Mapping — build a map for Nav2
=====================================
Phase 1 of Nav2 bring-up: drive the robot around with teleop while slam_toolbox
builds an occupancy grid from /scan + odom -> base_link. Save the result when done.

Nodes launched:
  1. kobuki_base_node — serial driver: /cmd_vel -> motors, encoders -> /odom + tf(odom->base_link)
  2. ldlidar_stl_ros2_node (LD19) — publishes /scan + static tf(base_link->base_laser)
  3. slam_toolbox (online_async) — builds map, publishes tf(map->odom)

Drive it yourself in another terminal (this launch does not start teleop):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard

How to run:
  ros2 launch botzilla_nav slam_mapping.launch.py
  ros2 launch botzilla_nav slam_mapping.launch.py lidar_port:=/dev/ttyUSB0

Once the map covers the area you need, save it:
  ros2 run nav2_map_server map_saver_cli -f ~/proactive-social-nav/ros2_ws/src/botzilla_nav/maps/botzilla_map

Then point nav2_bringup.launch.py's `map` argument at the resulting .yaml file.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    lidar_port = LaunchConfiguration('lidar_port')

    slam_params_file = os.path.join(
        get_package_share_directory('botzilla_nav'),
        'config',
        'slam_toolbox_params.yaml',
    )

    declare_lidar_port_cmd = DeclareLaunchArgument(
        'lidar_port', default_value='/dev/ldlidar',
        description='Serial port the LD19 lidar is connected to. Defaults to the '
                    'stable udev symlink from setup_udev.sh — raw /dev/ttyUSB* numbers '
                    'swap with the Kobuki between reboots and kill the lidar driver.',
    )

    kobuki_base = Node(
        package='botzilla_control',
        executable='kobuki_base_node',
        name='kobuki_base_node',
        output='screen',
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('ldlidar_stl_ros2'),
            '/launch/ld19.launch.py',
        ]),
        launch_arguments={'port_name': lidar_port}.items(),
    )

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file, {'use_sim_time': False}],
    )

    return LaunchDescription([
        declare_lidar_port_cmd,
        LogInfo(msg='[slam_mapping] ══════════════════════════════════════════'),
        LogInfo(msg='[slam_mapping] Drive the robot with teleop in another terminal:'),
        LogInfo(msg='[slam_mapping]   ros2 run teleop_twist_keyboard teleop_twist_keyboard'),
        LogInfo(msg='[slam_mapping] Watch progress in RViz2 (Fixed Frame: map, add Map display).'),
        LogInfo(msg='[slam_mapping] When done, save the map:'),
        LogInfo(msg='[slam_mapping]   ros2 run nav2_map_server map_saver_cli -f <path>/botzilla_map'),
        LogInfo(msg='[slam_mapping] ══════════════════════════════════════════'),
        kobuki_base,
        lidar_launch,
        slam_toolbox_node,
    ])
