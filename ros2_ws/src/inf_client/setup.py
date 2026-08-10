import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'inf_client'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'proto'), glob('proto/*.proto')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nilum',
    maintainer_email='sachithyanilum@gmail.com',
    description='gRPC client: forwards /scan and /odom from the robot to inf_server.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'grpc_client_node = inf_client.grpc_client_node:main',
        ],
    },
)
