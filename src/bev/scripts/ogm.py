#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# ROS messages
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from std_msgs.msg import Header

# Suppose this is your YOLO detection message
from yolo_msgs.msg import Detection, DetectionArray  # Adjust import path as needed
import numpy as np


class BoundingBoxOccupancyNode(Node):
    def __init__(self):
        super().__init__('bounding_box_occupancy_node')
        self.get_logger().info("Starting BoundingBoxOccupancyNode")

        # Subscriber to bounding box detections
        self.detection_sub = self.create_subscription(
            DetectionArray,
            '/yolo/detections_3d',    
            self.detections_callback,
            100
        )

        # Publisher for OccupancyGrid
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid,
            '/bev/occupancy',
            10
        )

        # Initialize a simple occupancy grid
        self.resolution = 0.1  # 10 cm per cell
        self.grid_width = 200  # number of cells in x
        self.grid_height = 200 # number of cells in y
        # The origin of the occupancy grid (bottom-left corner in world coordinates)
        self.origin_x = -10.0
        self.origin_y = -10.0

        # Homography matrix (3x3) - you must define or compute this
        # Example placeholder: identity matrix
        self.H = np.eye(3)

    def detections_callback(self, detection_msg: Detection):
        """
        Callback for each detection from YOLO. We extract the 3D box,
        project onto the ground, then fill the occupancy grid.
        """
        self.get_logger().info("Inside callback, processing detections")

        # Initialize an empty occupancy grid
        occupancy_grid = self.create_empty_grid()

        # Iterate through each detection in the list
        for detection in detection_msg.detections:
            # 1. Parse 3D bounding box info
            center_pose = detection.bbox3d.center  # geometry_msgs/Pose
            size = detection.bbox3d.size          # geometry_msgs/Vector3
            frame_id = detection.bbox3d.frame_id  # e.g. "odom"

            # Log the details of the detection (optional, for debugging)
            self.get_logger().info(
                f"Processing detection: Class={detection.class_name}, Score={detection.score}, "
                f"Center=({center_pose.position.x}, {center_pose.position.y}, {center_pose.position.z}), "
                f"Size=({size.x}, {size.y}, {size.z})"
            )

            # 2. Compute bounding box corner points (in 3D)
            corners_3d = self.compute_3d_corners(center_pose, size)

            # 3. Project corners onto ground plane
            corners_2d = self.project_corners_homography(corners_3d)

            # 4. Mark these corners on the occupancy grid
            # This function updates the occupancy grid data for each bounding box
            occupancy_grid.data = self.fill_occupancy_grid(occupancy_grid, corners_2d)

        # 5. Publish the updated occupancy grid
        self.occupancy_pub.publish(occupancy_grid)
        self.get_logger().info("Published updated occupancy grid")


    def compute_3d_corners(self, center_pose: Pose, size: Vector3):
        """
        Compute the 8 corners of the 3D bounding box relative to its center.
        This example assumes the bounding box is axis-aligned in the frame it is given.
        If there's orientation in center_pose.orientation, you must rotate these corners accordingly.
        """
        cx, cy, cz = center_pose.position.x, center_pose.position.y, center_pose.position.z
        dx, dy, dz = size.x, size.y, size.z

        # Half-dimensions
        hx, hy, hz = dx/2.0, dy/2.0, dz/2.0

        # 8 corners in local bounding-box coordinates
        corners = np.array([
            [ cx - hx, cy - hy, cz - hz ],
            [ cx - hx, cy - hy, cz + hz ],
            [ cx - hx, cy + hy, cz - hz ],
            [ cx - hx, cy + hy, cz + hz ],
            [ cx + hx, cy - hy, cz - hz ],
            [ cx + hx, cy - hy, cz + hz ],
            [ cx + hx, cy + hy, cz - hz ],
            [ cx + hx, cy + hy, cz + hz ],
        ])

        # TODO: If orientation is non-zero, rotate these corners around (cx,cy,cz).
        # For brevity, not shown here.

        return corners

    def project_corners_homography(self, corners_3d):
        """
        Project the 3D corners onto a 2D ground plane using a homography matrix H (3x3).
        Typically, you'd do something like:
             [u, v, w]^T = H * [X, Y, Z]^T
        and then (u'/w, v'/w) = the 2D pixel/plane coords.

        If your bounding box is already in an "odom" or "map" frame where Z=0 is the ground,
        you might just do corners_2d = (X, Y).
        """
        corners_2d = []
        for corner in corners_3d:
            X, Y, Z = corner[0], corner[1], corner[2]

            # Example: If Z ~ 0 is ground, a naive projection might be just (X, Y)
            # corners_2d.append([X, Y])

            # Or apply the homography (assuming [X, Y, 1] input and ignoring Z):
            # This is a simplified approach. In real usage, you need the correct H that accounts for Z as well.
            vec_3x1 = np.array([X, Y, 1.0])
            uvw = self.H @ vec_3x1
            u, v, w = uvw[0], uvw[1], uvw[2] if abs(uvw[2]) > 1e-9 else 1e-9
            corners_2d.append([u / w, v / w])
        return corners_2d

    def create_empty_grid(self):
        """
        Create a nav_msgs/OccupancyGrid with the correct metadata,
        filled initially with -1 (unknown).
        """
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'odom'  # or 'map', etc.

        # Grid metadata
        grid.info.resolution = self.resolution
        grid.info.width = self.grid_width
        grid.info.height = self.grid_height
        # OccupancyGrid origin: bottom-left corner in world coords
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        # Initialize data array
        num_cells = self.grid_width * self.grid_height
        grid.data = [-1] * num_cells  # -1 => unknown
        return grid

    def fill_occupancy_grid(self, grid, corners_2d):
        """
        Mark cells as "occupied" (e.g., 100) inside the projected bounding box polygon.
        A simplistic approach is to rasterize the polygon in grid coordinates.
        """
        data = list(grid.data)

        # Convert from world (u, v) => grid (col, row)
        def world_to_grid(u, v):
            col = int((u - self.origin_x) / self.resolution)
            row = int((v - self.origin_y) / self.resolution)
            return (col, row)

        # We’ll just mark the corners in this minimal example, 
        # but you should implement a polygon fill/rasterization.
        for (u, v) in corners_2d:
            col, row = world_to_grid(u, v)
            if 0 <= col < self.grid_width and 0 <= row < self.grid_height:
                index = row * self.grid_width + col
                data[index] = 100  # Mark cell as occupied

        return data


def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxOccupancyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
