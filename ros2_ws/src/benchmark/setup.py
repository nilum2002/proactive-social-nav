import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'benchmark'

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
    description='Instrumented inf_server variant: DR-SPAAM detection + Kalman tracking with latency/CPU/GPU CSV reporting.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'grpc_server_node = benchmark.grpc_server_node:main',
            'grpc_pipelined_node = benchmark.grpc_pipelined_node:main',
            'norfair_server_node = benchmark.norfair_server_node:main',
            'norfair_pipelined_node = benchmark.norfair_pipelined_node:main',
        ],
    },
)
