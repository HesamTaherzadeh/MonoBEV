import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    bev_package_dir = get_package_share_directory('bev')
    rviz_file_path = os.path.join(bev_package_dir, 'cfg', 'rviz.rviz')

    # Path to the YOLO launch file
    yolo_bringup_package_dir = get_package_share_directory('yolo_bringup')
    yolo_launch_file_path = os.path.join(yolo_bringup_package_dir, 'launch', 'yolov11.launch.py')

    return LaunchDescription([
        Node(
            package='bev',
            executable='bev_node',
            name='bev_node',
            output='screen',
            parameters=[os.path.join(bev_package_dir, 'cfg', 'cfg.yaml')]
        ),
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play',
                '/home/hesam/Desktop/datasets/kitti_raw/kitti_2011_10_03_drive_0027_synced'
            ],
            output='log'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(yolo_launch_file_path),
            launch_arguments={
                'input_image_topic': '/kitti/camera_color_left/image_raw'
            }.items()
        ),
        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_file_path],
            output='log'
        )
    ])
