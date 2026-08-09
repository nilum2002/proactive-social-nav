from setuptools import setup
import os
from glob import glob


package_name = 'nav2_coordinator'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
        glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
        glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nilum Mudaliarachchi',
    maintainer_email='Nilum@example.com',
    description='Thin ROS 2 service wrapper around Nav2 NavigateToPose action.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'nav2_coordinator_node = nav2_coordinator.nav2_coordinator_node:main',
        ],
    },
)