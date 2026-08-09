from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/ttyUSB0',
            description='Serial port for the LiDAR',
        ),
        DeclareLaunchArgument(
            'baudrate',
            default_value='230400',
            description='LiDAR serial baudrate',
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='laser',
            description='TF frame for LaserScan messages',
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='10.0',
            description='LiDAR publish rate in Hz',
        ),
        DeclareLaunchArgument(
            'parent_frame',
            default_value='base_link',
            description='Parent frame the LiDAR is mounted on',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='laser_static_tf',
            arguments=[
                '--x', '0.0', '--y', '0.0', '--z', '0.0',
                '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
                '--frame-id', LaunchConfiguration('parent_frame'),
                '--child-frame-id', LaunchConfiguration('frame_id'),
            ],
        ),
        Node(
            package='lidar_driver',
            executable='lidar_node',
            name='lidar_node',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'baudrate': LaunchConfiguration('baudrate'),
                'frame_id': LaunchConfiguration('frame_id'),
                'publish_rate': LaunchConfiguration('publish_rate'),
                'range_min': 0.05,
                'range_max': 8.0,
            }],
        ),
    ])
