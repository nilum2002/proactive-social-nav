import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import AndSubstitution, LaunchConfiguration, NotSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    kobuki_pkg_dir = get_package_share_directory('kobuki_driver')
    slam_config = os.path.join(kobuki_pkg_dir, 'config', 'slam_async_config.yaml')

    autostart = LaunchConfiguration('autostart')
    use_lifecycle_manager = LaunchConfiguration('use_lifecycle_manager')
    use_sim_time = LaunchConfiguration('use_sim_time')

    slam_node = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='',
        output='screen',
        parameters=[
            slam_config,
            {
                'use_lifecycle_manager': use_lifecycle_manager,
                'use_sim_time': use_sim_time,
            },
        ],
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager))),
    )

    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                LogInfo(msg='[LifecycleLaunch] slam_toolbox is activating.'),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_node),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                )),
            ],
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager))),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Transition slam_toolbox to active on startup. '
                        'Ignored when use_lifecycle_manager is true.',
        ),
        DeclareLaunchArgument(
            'use_lifecycle_manager',
            default_value='false',
            description='Let an external nav2 lifecycle manager drive the transitions instead',
        ),
        slam_node,
        configure_event,
        activate_event,
    ])
