from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

# Package path
pkg_typego = get_package_share_directory('typego')

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
    )
]

def generate_launch_description():
    slam_params_file = LaunchConfiguration('slam_params_file')
    rviz = LaunchConfiguration('rviz')

    slam_config_path = PathJoinSubstitution([pkg_typego, 'config', slam_params_file])
    rviz_config_path = PathJoinSubstitution([pkg_typego, 'rviz_cfgs', 'slam.rviz'])

    return LaunchDescription(ARGUMENTS + [
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                slam_config_path,
            ],
            remappings=[
                ('/scan', '/scan')
            ]
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
