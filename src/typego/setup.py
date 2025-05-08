from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'typego'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'rviz_cfgs'), glob(os.path.join('rviz_cfgs', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Guojun Chen',
    maintainer_email='guojun.chen@yale.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webui = typego.webui:main',
        ],
    },
)
