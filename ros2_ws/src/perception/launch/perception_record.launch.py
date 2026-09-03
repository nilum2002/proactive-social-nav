"""
perception_record.launch.py

Records a perception rosbag: camera + lidar + odometry + TF.

    ros2 launch perception perception_record.launch.py
    ros2 launch perception perception_record.launch.py bag_name:=corridor_run_1
    ros2 launch perception perception_record.launch.py topics:="/scan /odom"
    ros2 launch perception perception_record.launch.py use_camera:=false   # drivers already up

Storage is MCAP rather than the sqlite3 default - it appends without the
write-amplification sqlite3 has, which matters on an SD card, and an unclean
shutdown costs you the last chunk instead of the whole file.

Disk is the real limit.  /camera/image_raw (uncompressed) is deliberately absent
from the default topic list: at 640x480/15 it is ~14 MB/s against roughly 4.8 GB
free, which fills the card in about six minutes.  The compressed topic is
~1 MB/s.  The pre-flight check prints free space and estimated runtime first.
"""

import os
import shutil

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Compressed camera + lidar + odom + TF: everything needed to replay what the
# robot perceived, without anything that would blow the card.
DEFAULT_TOPICS = (
    '/camera/image_raw/compressed '
    '/camera/camera_info '
    '/scan '
    '/odom '
    '/tf '
    '/tf_static'
)

# Rough steady-state cost of DEFAULT_TOPICS, for the runtime estimate.
EST_MB_PER_S = 1.3


def _launch_setup(context, *args, **kwargs):
    """Resolve arguments, report disk headroom, and build the record command."""
    cfg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    output_dir = os.path.expanduser(cfg('output_dir'))
    bag_name = cfg('bag_name')
    storage = cfg('storage')
    max_bag_size = cfg('max_bag_size')
    compress = cfg('compress').lower() in ('true', '1', 'yes')
    topic_list = [t for t in cfg('topics').split() if t]

    os.makedirs(output_dir, exist_ok=True)

    usage = shutil.disk_usage(output_dir)
    free_gb = usage.free / 1e9
    minutes = (usage.free / 1e6) / EST_MB_PER_S / 60.0

    actions = [LogInfo(msg=(
        f'[record] {len(topic_list)} topic(s) -> {output_dir}/{bag_name} ({storage})'
    )), LogInfo(msg=(
        f'[record] {free_gb:.1f} GB free - roughly {minutes:.0f} min at '
        f'~{EST_MB_PER_S:.1f} MB/s for the default topic set'
    ))]

    if any(t.endswith('/image_raw') for t in topic_list):
        actions.append(LogInfo(msg=(
            '[record] WARNING: recording an uncompressed image topic. Expect '
            '~14 MB/s; this card holds only a few minutes of that.'
        )))

    if free_gb < 1.0:
        actions.append(LogInfo(msg=(
            '[record] WARNING: under 1 GB free. Free space or point output_dir '
            'at external storage before recording anything long.'
        )))

    cmd = [
        'ros2', 'bag', 'record',
        '--storage', storage,
        '--output', os.path.join(output_dir, bag_name),
    ]
    if max_bag_size and max_bag_size != '0':
        cmd += ['--max-bag-size', max_bag_size]
    if compress:
        # Per-file rather than per-message: one zstd pass over a finished file
        # costs far less CPU than compressing every message on a Pi.
        cmd += ['--compression-mode', 'file', '--compression-format', 'zstd']
    cmd += topic_list

    actions.append(ExecuteProcess(cmd=cmd, output='screen', shell=False))
    return actions


def generate_launch_description():
    perception_share = get_package_share_directory('perception')
    camera_launch = os.path.join(perception_share, 'launch', 'camera.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_name',
            default_value='perception_bag',
            description='Bag directory name, created inside output_dir',
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='~/bags',
            description='Where bags are written',
        ),
        DeclareLaunchArgument(
            'topics',
            default_value=DEFAULT_TOPICS,
            description='Space-separated topics to record',
        ),
        DeclareLaunchArgument(
            'storage',
            default_value='mcap',
            description='rosbag2 storage plugin (mcap or sqlite3)',
        ),
        DeclareLaunchArgument(
            'max_bag_size',
            default_value='2000000000',
            description='Start a new file at this many bytes (0 = never split)',
        ),
        DeclareLaunchArgument(
            'use_camera',
            default_value='true',
            description='Start camera_node as part of this launch',
        ),
        DeclareLaunchArgument(
            'compress',
            default_value='false',
            description=(
                'zstd-compress each finished bag file. JPEG is already '
                'compressed, so this mainly helps /scan-only bags.'
            ),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera_launch),
            condition=IfCondition(LaunchConfiguration('use_camera')),
        ),

        OpaqueFunction(function=_launch_setup),
    ])
