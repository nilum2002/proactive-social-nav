import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'inf_server_udp'

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
    description='UDP server: DR-SPAAM + Kalman tracking on single-datagram scans from inf_client_udp.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'udp_server_node = inf_server_udp.udp_server_node:main',
            'udp_pipelined_node = inf_server_udp.udp_pipelined_node:main',
        ],
    },
)
