import os
from glob import glob
from setuptools import setup

package_name = 'lidar_grpc_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'proto'), glob('proto/*.proto')),
    ],
    install_requires=['setuptools', 'grpcio', 'protobuf'],
    zip_safe=True,
    maintainer='nilum2002',
    maintainer_email='didulahirupama28@gmail.com',
    description='Streams ROS2 LaserScan over gRPC to an off-board consumer',
    license='MIT',
    entry_points={
        'console_scripts': [
            'grpc_lidar_server_node = lidar_grpc_bridge.grpc_lidar_server_node:main',
        ],
    },
)
