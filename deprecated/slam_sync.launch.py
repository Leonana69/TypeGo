import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, LogInfo,
                            RegisterEventHandler, GroupAction, 
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition,UnlessCondition
from launch.events import matches_action
from launch.substitutions import (AndSubstitution, LaunchConfiguration,
                                  NotSubstitution, PathJoinSubstitution)
from launch_ros.actions import LifecycleNode,PushRosNamespace,Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from nav2_common.launch import RewrittenYaml
from launch_ros.descriptions import ComposableNode, ParameterFile

pkg_cpsl_navigation = get_package_share_directory('typego')
pkg_slam_toolbox = get_package_share_directory('slam_toolbox')

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='false',
                          choices=['true', 'false'],
                          description='Use sim time'),
    DeclareLaunchArgument('use_lifecycle_manager', default_value='false',
                          choices=['true', 'false'],
                          description='Enable bond connection during node activation'),
    DeclareLaunchArgument('slam_params_file',
                          default_value='slam.yaml',
                          description='SLAM YAML file in the config folder'),
    DeclareLaunchArgument('rviz',
                          default_value='false',
                          choices=['true','false'],
                          description='Display an RViz window with navigation')
]

def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = "true"
    use_lifecycle_manager = LaunchConfiguration('use_lifecycle_manager')
    slam_params = LaunchConfiguration('slam_params_file')
    rviz = LaunchConfiguration('rviz')

    odom_frame = "odom"
    base_frame = "base_link"
    scan_topic_str = "scan"
    
    rviz_config_file = PathJoinSubstitution([pkg_cpsl_navigation, 'rviz_cfgs', 'slam.rviz'])

    slam_params_str = slam_params.perform(context)
    slam_config_file = PathJoinSubstitution([pkg_cpsl_navigation, 'config', slam_params_str])

    #substituting parameters
    param_substitutions = {
        'autostart': autostart,
        'scan_topic': scan_topic_str,
        'odom_frame': odom_frame,
        'base_frame': base_frame
    }

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=slam_config_file,
            root_key="",
            param_rewrites=param_substitutions,
            convert_types=True,
        ),
        allow_substs=True,
    )

    #slam toolbox setting up
    start_sync_slam_toolbox_node = LifecycleNode(
        parameters = [
          configured_params,
          {
            'use_lifecycle_manager': use_lifecycle_manager,
            'use_sim_time': use_sim_time
          }
        ],
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace=""
    )

    configure_event = EmitEvent(
        event = ChangeState(
            lifecycle_node_matcher = matches_action(start_sync_slam_toolbox_node),
            transition_id = Transition.TRANSITION_CONFIGURE
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager)))
    )

    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node = start_sync_slam_toolbox_node,
            start_state = "configuring",
            goal_state = "inactive",
            entities = [
                LogInfo(msg = "[LifecycleLaunch] Slamtoolbox node is activating."),
                EmitEvent(event = ChangeState(
                    lifecycle_node_matcher = matches_action(start_sync_slam_toolbox_node),
                    transition_id = Transition.TRANSITION_ACTIVATE
                ))
            ]
        ),
        condition = IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager)))
    )
    
    # Apply the following re-mappings only within this group
    slam = GroupAction([
        start_sync_slam_toolbox_node,
        configure_event,
        activate_event,
        # Launch RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(rviz)
        )
    ])

    return [slam]


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld