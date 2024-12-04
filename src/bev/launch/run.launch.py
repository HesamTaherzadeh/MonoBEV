import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory for the 'bev' package
    bev_package_dir = get_package_share_directory('bev')

    return LaunchDescription([
        # BEV Node
        Node(
            package='bev',  # Package name
            executable='bev_node',  # Executable name
            name='bev_node',  # Node name
            output='screen',  # Output to the screen
            parameters=[os.path.join(bev_package_dir, 'cfg', 'cfg.yaml')],  # Path to the configuration file
            remappings=[('/camera/image', '/camera/image_raw')]  # Topic remapping
        ),
        # Bag Playback Process
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play',
                '/home/hesam/Desktop/datasets/kitti_raw/kitti_2011_10_03_drive_0027_synced'
            ],
            output='log'  # Log output
        )
    ])
