import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    kobuki_pkg_dir = get_package_share_directory('kobuki_driver')
    lidar_pkg_dir = get_package_share_directory('lidar_driver')
    kinect_pkg_dir = get_package_share_directory('kinect_ros2')
    
    kobuki_ekf_launch = os.path.join(kobuki_pkg_dir, 'launch', 'kobuki_ekf_launch.py')
    lidar_launch = os.path.join(lidar_pkg_dir, 'launch', 'lidar_launch.py')
    slam_launch = os.path.join(kobuki_pkg_dir, 'launch', 'slam_launch.py')
    kinect_launch = os.path.join(kinect_pkg_dir, 'launch', 'kinect_stream_launch.py')


    return LaunchDescription([
        DeclareLaunchArgument(
            'kobuki_port',
            default_value='/dev/kobuki',
            description='Serial port for Kobuki',
        ),
        DeclareLaunchArgument(
            'lidar_port',
            default_value='/dev/lidar',
            description='Serial port for LiDAR',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),

        # Kobuki driver + EKF fusion
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(kobuki_ekf_launch),
            launch_arguments=[
                ('port', LaunchConfiguration('kobuki_port')),
                ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ],
        ),

        # Static transform: laser frame relative to base_link
        # Keep this available before SLAM starts so scans can be transformed immediately.
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
        
        # SLAM Toolbox
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments=[('use_sim_time', LaunchConfiguration('use_sim_time'))],
        ),
        
        # Kinect Stream
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(kinect_launch),
            launch_arguments=[('use_sim_time', LaunchConfiguration('use_sim_time'))],
        ),
    ])
