import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path definitions
    bev_package_dir = get_package_share_directory('bev')
    yolo_bringup_package_dir = get_package_share_directory('yolo_bringup')

    rviz_file_path = os.path.join(bev_package_dir, 'cfg', 'rviz.rviz')
    urdf_file_path = os.path.join(bev_package_dir, 'urdf', 'robot.urdf')
    ekf_config_file = os.path.join(bev_package_dir, 'cfg', 'cfg.yaml')
    yolo_launch_file_path = os.path.join(yolo_bringup_package_dir, 'launch', 'yolo.launch.py')

    # ROS 2 Launch Description
    return LaunchDescription([
        # Robot State Publisher for URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_file_path).read()}],
        ),

        # NavSat Transform Node (GPS -> Odometry)
        # Node(
        #     package='robot_localization',
        #     executable='navsat_transform_node',
        #     name='navsat_transform_node',
        #     output='screen',
        #     parameters=[{
        #         'publish_tf': True,
        #         'use_odometry_yaw': True,
        #         'wait_for_datum': False,
        #         'magnetic_declination_radians': 0.0,
        #         'yaw_offset': 0.0,
        #         'zero_altitude': True,
        #         'broadcast_utm_transform': False
        #     }],
        #     remappings=[
        #         ('imu/data', '/kitti/oxts/imu'),
        #         ('gps/fix', '/kitti/oxts/gps/fix'),
        #         ('gps/vel', '/kitti/oxts/gps/vel'),
        #         ('odometry/gps', '/odometry/gps'),
        #     ]
        # ),

        # # EKF Localization Node
        # Node(
        #     package='robot_localization',
        #     executable='ekf_node',
        #     name='ekf_filter_node',
        #     output='screen',
        #     parameters=[ekf_config_file]
        # ),

        # BEV Node
        Node(
            package='bev',
            executable='bev_node',
            name='bev_node',
            output='screen',
            parameters=[ekf_config_file]
        ),
        
        Node(
            package='bev',
            executable='ego.py',
            name='ego',
            output='screen'
        ),
        
        Node(
            package='bev',
            executable='ogm.py',
            name='ogm',
            output='screen'
            ),

         Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_static_tf',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'world']
        ),

        # Play the KITTI ROS 2 bag file
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', '-r', '1',
                '/home/hesam/Desktop/datasets/kitti_raw/kitti_2011_10_03_drive_0027_synced'
            ],
            output='log'
        ),

         TimerAction(
            period=5.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(yolo_launch_file_path),
                    launch_arguments={
                        'input_image_topic': '/rgb',
                        'use_3d': 'True',
                        'input_depth_topic': '/depth',
                        'input_depth_info_topic': '/camera_params',
                        'depth_image_units_divisor': '1'
                    }.items()
                )
            ]
        ),

         ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_file_path],
            output='log'
        )
    ])
