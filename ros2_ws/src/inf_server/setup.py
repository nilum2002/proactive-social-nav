import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'inf_server'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'proto'), glob('proto/*.proto')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nilum',
    maintainer_email='sachithyanilum@gmail.com',
    description='gRPC inference server: DR-SPAAM detection + Kalman tracking for a robot streaming /scan and /odom.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'grpc_server_node = inf_server.grpc_server_node:main',
            'grpc_pipelined_node = inf_server.grpc_pipelined_node:main',
        ],
    },
)
