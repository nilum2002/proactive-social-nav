import os
from glob import glob
from setuptools import setup

package_name = 'lidar_grpc_client'

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
    ],
    install_requires=['setuptools', 'grpcio', 'protobuf'],
    zip_safe=True,
    maintainer='nilum2002',
    maintainer_email='didulahirupama28@gmail.com',
    description='Receives LaserScan over gRPC and republishes it as a ROS2 topic',
    license='MIT',
    entry_points={
        'console_scripts': [
            'grpc_scan_client_node = lidar_grpc_client.grpc_scan_client_node:main',
        ],
    },
)
