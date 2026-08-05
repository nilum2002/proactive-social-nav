import os
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

_NORESET = os.path.join(
    os.path.expanduser('~'),
    'Desktop/Bozilla-ws/final-project-botzilla/noreset.so'
)


def generate_launch_description():
    kobuki_base = Node(
        package='botzilla_control',
        executable='kobuki_base_node',
        name='kobuki_base_node',
        output='screen',
    )

    kinect_bridge = Node(
        package='botzilla_perception',
        executable='kinect_bridge',
        name='kinect_bridge',
        output='screen',
        additional_env={'LD_PRELOAD': _NORESET},
    )

    yolo_node = Node(
        package='botzilla_perception',
        executable='yolo_node',
        name='yolo_node',
        output='screen',
        additional_env={'QT_QPA_PLATFORM': 'offscreen'},
    )

    apriltag_node = Node(
        package='botzilla_perception',
        executable='apriltag_node',
        name='apriltag_node',
        output='screen',
        additional_env={'QT_QPA_PLATFORM': 'offscreen'},
    )

    final_test = Node(
        package='botzilla_control',
        executable='final_test_node',
        name='final_test_node',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[final_test] ═══════════════════════════════════════════════'),
        LogInfo(msg='[final_test] BotZilla Final Integration Test (SSH/headless)'),
        LogInfo(msg='[final_test] Mission: collect cube → deliver to AprilTag → home'),
        LogInfo(msg='[final_test] Arena: 300 cm × 280 cm | Start: CENTRE facing LENGTH WALL'),
        LogInfo(msg='[final_test] Monitor: ros2 topic echo /mission/status'),
        LogInfo(msg='[final_test] Monitor: ros2 topic echo /odom'),
        LogInfo(msg='[final_test] ═══════════════════════════════════════════════'),
        kobuki_base,
        kinect_bridge,
        yolo_node,
        apriltag_node,
        final_test,
    ])
