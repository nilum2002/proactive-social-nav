import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from pathlib import Path
from launch.actions import TimerAction

def generate_launch_description():
    qbot_description_pkg_path = get_package_share_directory('qbot_description')
    world_file = os.path.join(qbot_description_pkg_path, 'sdf', 'world.sdf')
    urdf_path = Path(qbot_description_pkg_path) / 'urdf' / 'qbot.urdf'
    rviz_config = os.path.join(qbot_description_pkg_path, 'rviz', 'qbot.rviz')
    urdf_robot_description = urdf_path.read_text()
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description':  urdf_robot_description},
                    {'use_sim_time': True}
                    ],
        )
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    
    lidar_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
    )
    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'gz', 'sim', '-r', world_file
            ],
            output='screen'
        ),
        robot_state_publisher_node,
        rviz2,
        lidar_bridge_node,
        # Delay controller spawners to ensure Gazebo is ready
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster', '--ros-args', '-p', 'use_sim_time:=true'],
                    output='screen'
                ),
                ExecuteProcess(
                    cmd=['ros2', 'run', 'controller_manager', 'spawner', 'diff_drive_controller', '--ros-args', '-p', 'use_sim_time:=true'],
                    output='screen'
                ),
            ]
        ),
        # Bridging and remapping Gazebo topics to ROS 2 (replace with your own topics)
        # Node(
        #     package='ros_gz_bridge',
        #     executable='parameter_bridge',
        #     arguments=['/example_imu_topic@sensor_msgs/msg/Imu@gz.msgs.IMU',],
        #     remappings=[('/example_imu_topic',
        #                  '/remapped_imu_topic'),],
        #     output='screen'
        # ),
    ])