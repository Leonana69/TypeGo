from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

# Package path
pkg_typego_sdk = get_package_share_directory('typego_sdk')
print(get_package_share_directory('nav2_bringup'))

def generate_launch_description():
    nav2_config = os.path.join(
        pkg_typego_sdk,
        'config',
        'nav2_params.yaml'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('nav2_bringup'),
                '/launch/navigation_launch.py'
            ]),
            launch_arguments={
                'use_sim_time': 'false',
                'params_file': nav2_config,
                'autostart': 'true'
            }.items()
        )
    ])