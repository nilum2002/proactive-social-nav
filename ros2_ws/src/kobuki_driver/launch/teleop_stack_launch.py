import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    lidar_pkg_dir = get_package_share_directory('lidar_driver')
    inf_client_pkg_dir = get_package_share_directory('inf_client')

    lidar_launch = os.path.join(lidar_pkg_dir, 'launch', 'rplidar_c1_launch.py')
    inf_client_launch = os.path.join(inf_client_pkg_dir, 'launch', 'inf_client.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'kobuki_port',
            default_value='/dev/kobuki',
            description='Serial port for Kobuki',
        ),
        DeclareLaunchArgument(
            'lidar_port',
            # /dev/lidar does not exist; the udev rule creates /dev/ldlidar.
            # With the wrong path the driver throws on open and /scan is never
            # published, which downstream looks like "inf_client sends 0 frames".
            # The RPLIDAR C1M1 uses the same CP2102 bridge (10c4:ea60) as the
            # LD19 did, so the existing udev rule still matches and the symlink
            # name is now just historical.
            default_value='/dev/ldlidar',
            description='Serial port for LiDAR',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),

        # Kobuki base driver only -- teleop drives the robot directly, so raw
        # wheel odometry is enough and the EKF fusion node isn't needed.
        Node(
            package='kobuki_driver',
            executable='kobuki_node',
            name='kobuki_node',
            output='screen',
            parameters=[
                {'port': LaunchConfiguration('kobuki_port')},
                {'cmd_rate': 10},
                {'odom_pub_rate': 10},
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
        ),

        # Static transform: laser frame relative to base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '0.1',
                '0.0',
                '0.05',
                '0.0',
                '0.0',
                '0.0',
                'base_link',
                'laser',
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            name='laser_tf_publisher',
            output='screen',
        ),

        # LiDAR driver: RPLIDAR C1M1 via the vendored rplidar_ros package.
        # publish_tf is false because laser_tf_publisher above already owns
        # base_link->laser; letting both publish it would put two static
        # broadcasters on the same transform.
        #
        # This publishes 720 bins (5 kHz / 10 Hz -> 500 pts, which rplidar_ros
        # angle-compensates onto a 360*2 grid). That is fine over gRPC, which
        # streams over TCP. It is NOT fine for inf_client_udp: 720 bins is a
        # 1498 B datagram, past the 1472 B MTU, so every scan would IP-fragment
        # and a scan is lost if either fragment is. Resample to 450 before
        # putting this on the UDP path.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch),
            launch_arguments=[
                ('serial_port', LaunchConfiguration('lidar_port')),
                ('publish_tf', 'false'),
            ],
        ),

        # gRPC client: forwards /scan and /odom to inf_server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(inf_client_launch),
        ),
    ])
