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
        DeclareLaunchArgument(
            'mount_x',
            default_value='0.1',
            description='LiDAR mount offset forward of parent_frame, in metres',
        ),
        DeclareLaunchArgument(
            'mount_y',
            default_value='0.0',
            description='LiDAR mount offset left of parent_frame, in metres',
        ),
        DeclareLaunchArgument(
            'mount_z',
            default_value='0.05',
            description='LiDAR mount height above parent_frame, in metres',
        ),
        DeclareLaunchArgument(
            'mount_yaw',
            default_value='0.0',
            description='LiDAR yaw relative to parent_frame, in radians',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='laser_static_tf',
            arguments=[
                '--x', LaunchConfiguration('mount_x'),
                '--y', LaunchConfiguration('mount_y'),
                '--z', LaunchConfiguration('mount_z'),
                '--roll', '0.0', '--pitch', '0.0',
                '--yaw', LaunchConfiguration('mount_yaw'),
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
