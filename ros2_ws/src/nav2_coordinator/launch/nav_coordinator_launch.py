from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for the nav2_coordinator_node (debug, info, warn, error, fatal).',
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Optional ROS namespace for the node.',
    )

    nav2_coordinator_node = Node(
        package='nav2_coordinator',
        executable='nav2_coordinator_node',
        name='nav2_coordinator_node',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': False},
        ],
        remappings=[
            ('navigate_to_pose', 'navigate_to_pose'),
            ('navigate_to_coordinate', 'navigate_to_coordinate'),
        ],
    )

    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=nav2_coordinator_node,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return LaunchDescription([
        log_level_arg,
        namespace_arg,
        nav2_coordinator_node,
        shutdown_on_exit,
    ])