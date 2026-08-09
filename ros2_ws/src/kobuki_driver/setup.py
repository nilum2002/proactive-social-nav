import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'kobuki_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nilum Mudaliarachchi',
    maintainer_email='nilum@todo.todo',
    description='Kobuki ROS2 driver package',
    license='None',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kobuki_node = kobuki_driver.kobuki_node:main',
            'nav_action_server = kobuki_driver.nav_action_server:main',
            'kobuki_nav_server = kobuki_driver.kobuki_nav_server:main',
        ],
            

    },
)
