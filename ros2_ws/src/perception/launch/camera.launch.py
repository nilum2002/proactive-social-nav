"""
camera.launch.py

Just the camera publisher, under the /camera namespace:

    /camera/image_raw/compressed
    /camera/camera_info

Add the static base_link -> camera_link transform here once you have measured
where the webcam actually sits; without it nothing can place the images in the
robot frame, though bag recording works fine either way.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('perception')
    default_params = os.path.join(pkg_share, 'config', 'camera_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Camera parameter YAML',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='camera',
            description='Namespace for the camera topics',
        ),
        DeclareLaunchArgument(
            'publish_static_tf',
            default_value='false',
            description='Publish a base_link -> camera_link transform',
        ),
        DeclareLaunchArgument(
            'camera_x', default_value='0.1',
            description='Camera offset forward of base_link, metres'),
        DeclareLaunchArgument(
            'camera_y', default_value='0.0',
            description='Camera offset left of base_link, metres'),
        DeclareLaunchArgument(
            'camera_z', default_value='0.15',
            description='Camera height above base_link, metres'),

        Node(
            package='perception',
            executable='camera_node',
            name='camera_node',
            namespace=LaunchConfiguration('namespace'),
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('publish_static_tf')),
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_static_tf',
            arguments=[
                '--x', LaunchConfiguration('camera_x'),
                '--y', LaunchConfiguration('camera_y'),
                '--z', LaunchConfiguration('camera_z'),
                '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'camera_link',
            ],
        ),
    ])
