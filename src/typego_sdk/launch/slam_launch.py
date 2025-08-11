from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition

import os

# Package path
pkg_typego_sdk = get_package_share_directory('typego_sdk')

# Declare launch arguments
ARGUMENTS = [
    DeclareLaunchArgument(
        'slam_params_file',
        default_value='slam.yaml',
        description='SLAM YAML file in the config folder'
    ),
    DeclareLaunchArgument(
        'rviz',
        default_value='false',
        choices=['true', 'false'],
        description='Display an RViz window with SLAM config'
    ),
    DeclareLaunchArgument(
        'use_existing_map',
        default_value='true',
        choices=['true', 'false'],
        description='Use existing map'
    ),
]

def generate_launch_description():
    slam_params_file = LaunchConfiguration('slam_params_file')
    rviz = LaunchConfiguration('rviz')
    use_existing_map = LaunchConfiguration('use_existing_map')

    slam_config_path = PathJoinSubstitution([pkg_typego_sdk, 'config', slam_params_file])
    rviz_config_path = PathJoinSubstitution([pkg_typego_sdk, 'rviz_config', 'slam.rviz'])

    resource_dir = os.getenv('RESOURCE_DIR', '/home/guojun/Documents/Go2-Livox-ROS2/src/typego_sdk/resource/')
    map_params = {
        'map_file_name': os.path.join(resource_dir, '4th'),
        'map_start_pose': [0.2, -0.2, 0.0]
    }

    return LaunchDescription(ARGUMENTS + [
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_config_path, map_params],
            remappings=[('/scan', '/scan')],
            condition=IfCondition(use_existing_map)
        ),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_config_path],
            remappings=[('/scan', '/scan')],
            condition=UnlessCondition(use_existing_map)
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen',
            condition=IfCondition(rviz)
        )
    ])
