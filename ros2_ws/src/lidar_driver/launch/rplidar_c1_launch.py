"""RPLIDAR C1M1 on this robot: driver + the base_link->laser mount transform.

Wraps rplidar_ros's own rplidar_c1_launch.py rather than editing it, so the
vendored tree stays exactly as upstream shipped it. Mount arguments mirror
lidar_launch.py (the LD19 launch) so swapping sensors does not move the frame.

Measured on this unit (firmware 1.02, hardware rev 18):
    Standard mode, 5 kHz sample rate, 10 Hz, max_distance 16.0 m
    -> 500 points/revolution, which rplidar_ros's angle_compensate bins into
       360 * 2 = 720 fixed slots at 0.499 deg.

NOTE: angle_compensate defaults to FALSE here. Compensated, the sweep becomes
720 fixed bins -> a 1498 B datagram, past the 1472 B MTU, so inf_client_udp
IP-fragmented every scan (measured; a fragmented scan is lost if either half
is). Raw, the C1 emits its native ~500 points -> ~1058 B, comfortably inside
the 1400 B budget.

The cost: a raw sweep's point count varies slightly scan to scan. DR-SPAAM on
inf_server sizes its FOV once from the first scan it sees, so if it starts
rejecting scans or mis-sizing the FOV, that is this trade-off, not a bug --
re-enable with angle_compensate:=true and resample to 450 bins instead.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_port', default_value='/dev/ldlidar',
            description='Serial port. The C1M1 uses the same CP2102 bridge '
                        '(10c4:ea60) as the LD19, so the existing udev symlink matches.'),
        DeclareLaunchArgument(
            'serial_baudrate', default_value='460800',
            description='C1M1 runs at 460800, not the LD19 230400'),
        DeclareLaunchArgument('frame_id', default_value='laser'),
        DeclareLaunchArgument(
            'scan_mode', default_value='Standard',
            description='Standard = 5 kHz / 10 Hz / 16 m on this unit'),
        DeclareLaunchArgument(
            'inverted', default_value='false',
            description='The LD19 driver needed invert_angle:=true for REP-103; '
                        'rplidar_ros already reports counter-clockwise. Flip this '
                        'only if the map comes out mirrored.'),
        DeclareLaunchArgument(
            'angle_compensate', default_value='false',
            description='false = publish the native ~500-point sweep, which fits one '
                        'UDP datagram. true = bin onto a fixed 720-slot grid, which '
                        'gives DR-SPAAM a constant scan length but overflows the MTU; '
                        'only pair true with a resampler down to 450 bins.'),
        DeclareLaunchArgument(
            'publish_tf', default_value='true',
            description='Publish base_link->laser here. Set false when including '
                        'this from a stack launch that already publishes it, so '
                        'the transform has exactly one owner.'),
        DeclareLaunchArgument('parent_frame', default_value='base_link'),
        DeclareLaunchArgument('mount_x', default_value='0.1'),
        DeclareLaunchArgument('mount_y', default_value='0.0'),
        DeclareLaunchArgument('mount_z', default_value='0.05'),
        DeclareLaunchArgument('mount_yaw', default_value='0.0'),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='laser_static_tf',
            condition=IfCondition(LaunchConfiguration('publish_tf')),
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
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            output='screen',
            parameters=[{
                'channel_type': 'serial',
                'serial_port': LaunchConfiguration('serial_port'),
                'serial_baudrate': LaunchConfiguration('serial_baudrate'),
                'frame_id': LaunchConfiguration('frame_id'),
                'inverted': LaunchConfiguration('inverted'),
                'angle_compensate': LaunchConfiguration('angle_compensate'),
                'scan_mode': LaunchConfiguration('scan_mode'),
            }],
        ),
    ])
