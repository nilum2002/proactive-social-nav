"""
Full navigation stack for the qbot, without EKF/IMU fusion.

Brings up, in order:
  kobuki_node        - /cmd_vel in, /odom out, broadcasts odom -> base_link
  lidar_node         - /scan, plus the static base_link -> laser transform
  map -> odom        - slam_toolbox (slam:=true) or map_server + amcl (slam:=false)
  nav2               - planner/controller/bt_navigator (navigation_launch.py)
  nav2_coordinator   - 'navigate_to_coordinate' service wrapping NavigateToPose

Two modes, selected by the `slam` argument:

  slam:=true   mapping.  slam_toolbox builds the grid live.  Drive with
               teleop_twist_keyboard, then save with:
                 ros2 run nav2_map_server map_saver_cli -f <repo>/maps/lab

  slam:=false  (default) navigation.  map_server serves the saved grid as the
               costmaps' static_layer; amcl localizes against it.  Objects that
               appeared after mapping are picked up by the obstacle_layer from
               live /scan, so the planner still routes around them without the
               saved map being touched.

amcl is configured with set_initial_pose at the map origin (there is no RViz
"2D Pose Estimate" over SSH), so start the robot on the spot where mapping began.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    kobuki_share = get_package_share_directory('kobuki_driver')
    lidar_share = get_package_share_directory('lidar_driver')
    coordinator_share = get_package_share_directory('nav2_coordinator')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    default_params = os.path.join(coordinator_share, 'config', 'nav2_params.yaml')
    # Written by `map_saver_cli -f <repo>/maps/lab`.  Override with map:=<path>;
    # nav2_stack.sh passes the repo-relative path so a moved checkout still works.
    default_map = os.path.join(
        os.path.expanduser('~'), 'proactive-social-nav', 'maps', 'lab.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    slam = LaunchConfiguration('slam')

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
            'slam',
            default_value='false',
            description='true  = build the map live with slam_toolbox (phase 1). '
                        'false = localize against the saved map with map_server '
                        '+ amcl (phase 2).',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Occupancy grid YAML to localize against. Ignored when slam:=true.',
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

        
        # Phase 1: SLAM builds map -> odom while mapping.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(kobuki_share, 'launch', 'slam_launch.py')),
            launch_arguments=[('use_sim_time', use_sim_time)],
            condition=IfCondition(slam),
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

        # Phase 2: map_server + amcl provide map -> odom from the saved map.
        # Must come after nav2_container: in composed mode this loads into it.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_share, 'launch', 'localization_launch.py')),
            launch_arguments=[
                ('map', LaunchConfiguration('map')),
                ('use_sim_time', use_sim_time),
                ('params_file', params_file),
                ('autostart', 'true'),
                ('use_composition', LaunchConfiguration('use_composition')),
                ('container_name', 'nav2_container'),
            ],
            condition=UnlessCondition(slam),
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
