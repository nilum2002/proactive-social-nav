import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    lidar_pkg_dir = get_package_share_directory('lidar_driver')
    inf_client_pkg_dir = get_package_share_directory('inf_client')

    lidar_launch = os.path.join(lidar_pkg_dir, 'launch', 'lidar_launch.py')
    inf_client_launch = os.path.join(inf_client_pkg_dir, 'launch', 'inf_client.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'kobuki_port',
            default_value='/dev/kobuki',
            description='Serial port for Kobuki',
        ),
        DeclareLaunchArgument(
            'lidar_port',
            # /dev/lidar does not exist; the udev rule creates /dev/ldlidar.
            # With the wrong path the driver throws on open and /scan is never
            # published, which downstream looks like "inf_client sends 0 frames".
            default_value='/dev/ldlidar',
            description='Serial port for LiDAR',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),

        # Kobuki base driver only -- teleop drives the robot directly, so raw
        # wheel odometry is enough and the EKF fusion node isn't needed.
        Node(
            package='kobuki_driver',
            executable='kobuki_node',
            name='kobuki_node',
            output='screen',
            parameters=[
                {'port': LaunchConfiguration('kobuki_port')},
                {'cmd_rate': 10},
                {'odom_pub_rate': 10},
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
        ),

        # Static transform: laser frame relative to base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '0.1',
                '0.0',
                '0.05',
                '0.0',
                '0.0',
                '0.0',
                'base_link',
                'laser',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            name='laser_tf_publisher',
            output='screen',
        ),

        # LiDAR driver
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch),
            launch_arguments=[
                ('serial_port', LaunchConfiguration('lidar_port')),
                ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ],
        ),

        # gRPC client: forwards /scan and /odom to inf_server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(inf_client_launch),
        ),
    ])
