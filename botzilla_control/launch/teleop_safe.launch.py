"""
Teleop with safety arbitration
================================
Launches the Kobuki driver + twist_mux, so keyboard teleop always overrides any
autonomous command (Nav2, once wired in — see cmd_vel_nav below, unused for now).

Nodes launched:
  1. kobuki_base_node — serial driver: /cmd_vel -> Kobuki wheel commands,
                         encoders -> /odom + tf(odom->base_link)
  2. twist_mux         — arbitrates cmd_vel_teleop (priority 100) and cmd_vel_nav
                         (priority 10, reserved for Nav2) onto /cmd_vel

teleop_twist_keyboard needs a live TTY for keypresses, so it is NOT launched here —
run it yourself in its own terminal, with its output remapped to cmd_vel_teleop:

  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_teleop

How to run:
  ros2 launch botzilla_control teleop_safe.launch.py

Monitor in a second terminal:
  ros2 topic echo /odom             # velocity + pose feedback
  ros2 run tf2_ros tf2_echo odom base_link
  ros2 topic echo /cmd_vel          # arbitrated output actually driving the motors
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'botzilla_control'

    config_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'twist_mux.yaml'
    )

    kobuki_base = Node(
        package=package_name,
        executable='kobuki_base_node',
        name='kobuki_base_node',
        output='screen',
    )

    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen',
        parameters=[config_path],
        remappings=[('cmd_vel_out', 'cmd_vel')],
    )

    return LaunchDescription([
        LogInfo(msg='[teleop_safe] ══════════════════════════════════════════'),
        LogInfo(msg='[teleop_safe] Run teleop yourself in another terminal:'),
        LogInfo(msg='[teleop_safe]   ros2 run teleop_twist_keyboard teleop_twist_keyboard \\'),
        LogInfo(msg='[teleop_safe]       --ros-args -r cmd_vel:=cmd_vel_teleop'),
        LogInfo(msg='[teleop_safe] ══════════════════════════════════════════'),
        kobuki_base,
        twist_mux,
    ])
