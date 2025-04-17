# Go2 Livox ROS2
A ros2 node that receives cloud points and state UDP stream from the Go2 native controller (RK3588 controller, not the external Orin).

## Installation
This package is built with ROS2 humble and colcon.
```bash
cd Go2-Livox-ROS2
colcon build
# or setup.zsh
source ./install/setup.sh
ros2 run livox_udp_receiver livox_udp_receiver_node
```