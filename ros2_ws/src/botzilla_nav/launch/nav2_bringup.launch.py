"""
Nav2 Bring-up — full path planning + obstacle avoidance
===========================================================
Phase 2 of Nav2 bring-up: localize against a saved map and run the full Nav2
stack (AMCL, costmaps, planner, controller, behavior tree navigator). This is
what nav2_coordinator's NavigateToPose action client talks to.

Nodes launched:
  1. kobuki_base_node — serial driver: /cmd_vel -> motors, encoders -> /odom + tf(odom->base_link)
  2. ldlidar_stl_ros2_node (LD19) — publishes /scan + static tf(base_link->base_laser)
  3. twist_mux — arbitrates cmd_vel_teleop (priority 100, manual override) and
                 cmd_vel_nav (priority 10, Nav2's output) onto /cmd_vel.
                 nav2_bringup's navigation_launch.py already remaps the
                 controller/velocity_smoother/collision_monitor output to
                 cmd_vel_nav, so this just works.
  4. nav2_bringup (bringup_launch.py) — map_server + amcl + planner_server +
                 controller_server + behavior_server + bt_navigator + ...

Two modes:
  - Localization (default): AMCL against a map built earlier with
    slam_mapping.launch.py (see that file's docstring). Best once you have a
    stable map — more accurate localization, lower CPU than live SLAM.
  - Live SLAM (slam:=True): slam_toolbox builds the map on the fly while Nav2
    plans/avoids obstacles against it. No pre-saved map needed — good for a
    first run or a space that changes. Costs more CPU than AMCL.

  NOTE: capitalized True/False, not lowercase. nav2_bringup's bringup_launch.py
  feeds `slam` straight into a PythonExpression (eval'd as Python code) to pick
  between its SLAM and localization branches — lowercase "true"/"false" raises
  `NameError: name 'true' is not defined` since Python's literals are capitalized.

How to run:
  # Localization against a saved map:
  ros2 launch botzilla_nav nav2_bringup.launch.py map:=/path/to/botzilla_map.yaml

  # Live SLAM, no map required:
  ros2 launch botzilla_nav nav2_bringup.launch.py slam:=True

  ros2 launch botzilla_nav nav2_bringup.launch.py map:=/path/to/botzilla_map.yaml lidar_port:=/dev/ttyUSB0

Then, from a workstation on the same ROS_DOMAIN_ID:
  rviz2 -d /opt/ros/jazzy/share/nav2_bringup/rviz/nav2_default_view.rviz
  - Set "2D Pose Estimate" to the robot's actual starting pose.
  - Set a "2D Nav Goal" to test planning + obstacle avoidance.

Once goals succeed in RViz, nav2_coordinator_node's navigate_to_coordinate
service will use this same NavigateToPose action.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_yaml_file = LaunchConfiguration('map')
    lidar_port = LaunchConfiguration('lidar_port')
    slam = LaunchConfiguration('slam')

    nav2_params_file = os.path.join(
        get_package_share_directory('botzilla_nav'),
        'config',
        'nav2_params.yaml',
    )
    twist_mux_config = os.path.join(
        get_package_share_directory('botzilla_control'),
        'config',
        'twist_mux.yaml',
    )

    declare_map_cmd = DeclareLaunchArgument(
        'map', default_value='',
        description='Full path to a map yaml file saved by slam_mapping.launch.py. '
                     'Ignored (and not required) when slam:=True.',
    )
    declare_lidar_port_cmd = DeclareLaunchArgument(
        'lidar_port', default_value='/dev/ldlidar',
        description='Serial port the LD19 lidar is connected to. Defaults to the '
                    'stable udev symlink from setup_udev.sh — raw /dev/ttyUSB* numbers '
                    'swap with the Kobuki between reboots and kill the lidar driver.',
    )
    declare_slam_cmd = DeclareLaunchArgument(
        'slam', default_value='False',
        description='True = build the map live with slam_toolbox while navigating '
                     '(no map required). False = localize with AMCL against `map`. '
                     'Must be capitalized True/False (nav2_bringup evals this as Python).',
    )
    declare_grpc_bridge_cmd = DeclareLaunchArgument(
        'grpc_bridge', default_value='true',
        description='Run lidar_grpc_bridge alongside Nav2, streaming /scan + /odom to '
                    'the remote inference PC. Costs ~25%% of one core and extra DDS '
                    'traffic — set false when tuning/debugging navigation on the Pi, '
                    'which is already CPU-bound running Nav2 + live SLAM.',
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

    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen',
        parameters=[twist_mux_config],
        remappings=[('cmd_vel_out', 'cmd_vel')],
    )

    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nav2_bringup'),
            '/launch/bringup_launch.py',
        ]),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': 'false',
            'params_file': nav2_params_file,
            'autostart': 'true',
            'slam': slam,
        }.items(),
    )
    # Streams /scan + /odom out to the inference PC over gRPC rather than raw DDS.
    # Node name must stay 'lidar_grpc_server_node' — that is the top-level key in
    # lidar_grpc_bridge's params.yaml, and params silently fail to load if it differs.
    grpc_bridge_config = os.path.join(
        get_package_share_directory('lidar_grpc_bridge'),
        'config',
        'params.yaml',
    )

    grpc_bridge_node = Node(
        package='lidar_grpc_bridge',
        executable='grpc_lidar_server_node',
        name='lidar_grpc_server_node',
        output='screen',
        parameters=[grpc_bridge_config],
        condition=IfCondition(LaunchConfiguration('grpc_bridge')),
    )

    return LaunchDescription([
        declare_map_cmd,
        declare_lidar_port_cmd,
        declare_slam_cmd,
        declare_grpc_bridge_cmd,
        LogInfo(msg='[nav2_bringup] ══════════════════════════════════════════'),
        LogInfo(msg='[nav2_bringup] Full Nav2 stack: AMCL + costmaps + planner + controller'),
        LogInfo(msg='[nav2_bringup] Set an initial pose in RViz2 before sending goals.'),
        LogInfo(msg='[nav2_bringup] Manual override still works: '),
        LogInfo(msg='[nav2_bringup]   ros2 run teleop_twist_keyboard teleop_twist_keyboard \\'),
        LogInfo(msg='[nav2_bringup]       --ros-args -r cmd_vel:=cmd_vel_teleop'),
        LogInfo(msg='[nav2_bringup] ══════════════════════════════════════════'),
        kobuki_base,
        lidar_launch,
        twist_mux,
        nav2_bringup_launch,
        grpc_bridge_node,
    ])
