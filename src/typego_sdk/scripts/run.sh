#!/bin/bash

# Source the ROS 2 setup file and run the nodes in the background
iox-roudi &
source /workspace/install/setup.bash && ros2 run typego_sdk livox_udp_receiver_node &
source /workspace/install/setup.bash && ros2 run typego_sdk go2_tf_service_node &
source /workspace/install/setup.bash && ros2 run typego_sdk gstreamer_receiver_node &
source /workspace/install/setup.bash && ros2 launch typego_sdk slam_launch.py &
# Wait until /map or /tf is available (example topic)
echo "⏳ Waiting for /map topic to be available..."
while ! ros2 topic list | grep -q "/map"; do
  sleep 1
done
source /workspace/install/setup.bash && ros2 run typego_sdk waypoints_node &
source /workspace/install/setup.bash && ros2 launch typego_sdk nav2_launch.py
