import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

# Paths
WORKSPACE  = '/home/nilum/proactive-social-nav'
VENV_PYTHON = os.path.join(WORKSPACE, 'venv', 'bin', 'python3')
DR_SPAAM   = os.path.join(WORKSPACE, 'dr_spaam')
VENV_SITE  = os.path.join(WORKSPACE, 'venv', 'lib', 'python3.12', 'site-packages')

def generate_launch_description():
    package_name = 'dr_spaam_ros2'

    config_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'params.yaml'
    )

    # Build PYTHONPATH: venv site-packages + dr_spaam source
    extra_pythonpath = DR_SPAAM + ':' + VENV_SITE
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    full_pythonpath = extra_pythonpath + (':' + current_pythonpath if current_pythonpath else '')

    node_script = os.path.join(
        get_package_share_directory(package_name).replace('/share/' + package_name, ''),
        '..', '..', 'lib', package_name, 'dr_spaam_ros2_node'
    )

    return LaunchDescription([
        Node(
            package=package_name,
            executable='dr_spaam_ros2_node',
            name='dr_spaam_ros2_node',
            output='screen',
            parameters=[config_path],
            # Override the interpreter by setting PYTHONPATH and using the prefix trick
            additional_env={
                'PYTHONPATH': full_pythonpath,
            },
            prefix=VENV_PYTHON + ' ',
        )
    ])
