from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',  # Correct executable name
            name='slam_toolbox',
            output='screen',
            parameters=[
                {'use_sim_time': False},  # Disable for real robot
                {'odom_frame': 'odom'},
                {'base_frame': 'base_link'},
                {'map_frame': 'map'},
                {'max_laser_range': 12.0},  # Adjust based on your LIDAR
                {'async_mode': True}  # Enable asynchronous mode
            ],
            remappings=[
                ('/scan', '/scan')  # Replace with your LIDAR topic
            ]
        )
    ])