#!/bin/bash

# Source the ROS 2 setup file and run the nodes in the background
iox-roudi &
source /workspace/install/setup.bash && ros2 run typego_sdk livox_udp_receiver_node &
source /workspace/install/setup.bash && ros2 run typego_sdk go2_tf_service_node &
source /workspace/install/setup.bash && ros2 run typego_sdk gstreamer_receiver_node &
source /workspace/install/setup.bash && ros2 launch typego_sdk slam_launch.py &
# source /workspace/install/setup.bash && ros2 run typego_sdk waypoints_node &
source /workspace/install/setup.bash && ros2 launch typego_sdk nav2_launch.py
