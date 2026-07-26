import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction
)

from launch_ros.actions import Node


def generate_launch_description():

    pkg_path = get_package_share_directory('qbot_description')

    urdf_file = os.path.join(pkg_path, 'urdf', 'qbot.urdf')
    world_file = os.path.join(pkg_path, 'sdf', 'qbot_world.sdf')
    rviz_config = os.path.join(pkg_path, 'rviz', 'qbot.rviz')

    # Read URDF
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    # Gazebo plugin discovery
    gazebo_plugin_path = os.environ.get('LD_LIBRARY_PATH', '')

    set_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=gazebo_plugin_path
    )

    set_ld_path = SetEnvironmentVariable(
        name='LD_LIBRARY_PATH',
        value=gazebo_plugin_path
    )

    # Start Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world_file],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': True}
        ],
        output='screen'
    )
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )
    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    # Controller spawners
    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller'],
        output='screen'
    )

    # Delay controllers (Gazebo must start first)
    delayed_joint_state = TimerAction(
        period=5.0,
        actions=[joint_state_spawner]
    )

    delayed_diff_drive = TimerAction(
        period=7.0,
        actions=[diff_drive_spawner]
    )

    # Lidar bridge
    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '--ros-args', '-r', '/lidar:=/scan'
        ],
        output='screen'
    )

    return LaunchDescription([
        set_plugin_path,
        set_ld_path,
        gazebo,
        robot_state_publisher,
        clock_bridge,
        rviz,
        lidar_bridge,
        delayed_joint_state,
        delayed_diff_drive
    ])