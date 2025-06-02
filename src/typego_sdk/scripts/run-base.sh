#!/bin/bash

# Source the ROS 2 setup file and run the nodes in the background
source /workspace/install/setup.bash && ros2 run typego_sdk livox_udp_receiver_node &
source /workspace/install/setup.bash && ros2 run typego_sdk gstreamer_receiver_node