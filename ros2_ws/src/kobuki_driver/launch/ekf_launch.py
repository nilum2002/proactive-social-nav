import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the kobuki_driver package directory
    kobuki_pkg_dir = get_package_share_directory('kobuki_driver')
    config_file = os.path.join(kobuki_pkg_dir, 'config', 'ekf_config.yaml')

    # Declare launch arguments (optional)
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time (set to true for bag playback)',
        ),
        
        # robot_localization EKF node
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[config_file, {'use_sim_time': use_sim_time}],
            remappings=[
                ('/odometry/filtered', '/odometry/filtered'),
            ],
            arguments=['--ros-args', '--log-level', 'info'],
        ),
    ])
