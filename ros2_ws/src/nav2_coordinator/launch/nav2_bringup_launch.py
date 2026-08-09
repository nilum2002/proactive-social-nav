"""
Full navigation stack for the qbot, without EKF/IMU fusion.

Brings up, in order:
  kobuki_node        - /cmd_vel in, /odom out, broadcasts odom -> base_link
  lidar_node         - /scan, plus the static base_link -> laser transform
  slam_toolbox       - builds the map live, broadcasts map -> odom
  nav2               - planner/controller/bt_navigator (navigation_launch.py)
  nav2_coordinator   - 'navigate_to_coordinate' service wrapping NavigateToPose

"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    kobuki_share = get_package_share_directory('kobuki_driver')
    lidar_share = get_package_share_directory('lidar_driver')
    coordinator_share = get_package_share_directory('nav2_coordinator')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    default_params = os.path.join(coordinator_share, 'config', 'nav2_params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'kobuki_port',
            default_value='/dev/kobuki',
            description='Serial port for the Kobuki base',
        ),
        DeclareLaunchArgument(
            'lidar_port',
            default_value='/dev/ldlidar',
            description='Serial port for the LiDAR',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Nav2 parameters YAML',
        ),
        DeclareLaunchArgument(
            'use_coordinator',
            default_value='true',
            description='Start the navigate_to_coordinate service wrapper',
        ),
        DeclareLaunchArgument(
            # Nav2 defaults this to False (one process per node). On a Pi 4 the
            # single-container form saves enough RAM to matter; flip it back to
            # False to isolate a node that is crashing.
            'use_composition',
            default_value='True',
            description='Load the Nav2 nodes into one composed container',
        ),

        
        Node(
            package='kobuki_driver',
            executable='kobuki_node',
            name='kobuki_node',
            output='screen',
            parameters=[{
                'port': LaunchConfiguration('kobuki_port'),
                'cmd_timeout': 0.5,
                'cmd_rate': 10,
                'odom_pub_rate': 30,
                'publish_odom_tf': True,
                'use_sim_time': use_sim_time,
            }],
        ),

        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(lidar_share, 'launch', 'lidar_launch.py')),
            launch_arguments=[
                ('serial_port', LaunchConfiguration('lidar_port')),
                ('frame_id', 'laser'),
                ('parent_frame', 'base_link'),
                ('use_sim_time', use_sim_time),
            ],
        ),

        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(kobuki_share, 'launch', 'slam_launch.py')),
            launch_arguments=[('use_sim_time', use_sim_time)],
        ),

        
        # Container for the composed form.  navigation_launch.py only calls
        # LoadComposableNodes against an existing container - it never creates
        # one (only bringup_launch.py does), so without this the composed path
        # loads nothing at all and fails silently.
        Node(
            condition=IfCondition(LaunchConfiguration('use_composition')),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[params_file, {'autostart': True}],
            output='screen',
        ),

        # Nav2 planner, controller and behaviour tree.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_share, 'launch', 'navigation_launch.py')),
            launch_arguments=[
                ('use_sim_time', use_sim_time),
                ('params_file', params_file),
                ('autostart', 'true'),
                ('use_composition', LaunchConfiguration('use_composition')),
                ('container_name', 'nav2_container'),
            ],
        ),

        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(coordinator_share, 'launch', 'nav_coordinator_launch.py')),
            condition=IfCondition(LaunchConfiguration('use_coordinator')),
        ),
    ])
