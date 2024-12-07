#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from bev_interface.msg import Homography
from yolo_msgs.msg import DetectionArray
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

class BEVOccupancyNode(Node):
    def __init__(self):
        super().__init__('bev_occupancy_node')
        print("initiated")
        
        # Subscribers
        self.homography_sub = self.create_subscription(
            Homography,
            '/homography',
            self.homography_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            DetectionArray,
            '/yolo/detections',
            self.detection_callback,
            10
        )
        
        # Publisher
        self.occ_pub = self.create_publisher(OccupancyGrid, '/bev/occupancy_grid', 10)
        
        # Store homography
        self.homography_matrix = None
        
        # Define Occupancy Grid parameters
        self.grid_width = 100
        self.grid_height = 100
        self.grid_resolution = 0.1  # meters per cell
        self.grid_origin_x = 0.0
        self.grid_origin_y = 0.0
        
        # Initialize occupancy grid data
        self.occupancy_data = [0] * (self.grid_width * self.grid_height)
        
        # Timer to periodically publish occupancy grid
        self.timer = self.create_timer(1.0, self.publish_occupancy_grid)

    def homography_callback(self, msg: Homography):
        # Extract 3x3 matrix
        print("Recieved homography")
        mat = np.array(msg.matrix).reshape((3,3))
        self.homography_matrix = mat
        self.get_logger().info("Homography matrix updated.")

    def detection_callback(self, msg):
        print("Recieved Detection")
        if self.homography_matrix is None:
            self.get_logger().warn("No homography received yet; cannot process detection.")
            return

        # Extract bounding box corners in image coordinates
        # Assuming bbox has fields: x, y, width, height (top-left origin)
        x = msg.bbox.center.x - (msg.bbox.size_x/2.0)
        y = msg.bbox.center.y - (msg.bbox.size_y/2.0)
        w = msg.bbox.size_x
        h = msg.bbox.size_y

        corners = [
            [x, y],            # top-left
            [x + w, y],        # top-right
            [x, y + h],        # bottom-left
            [x + w, y + h]     # bottom-right
        ]

        # Transform each corner using the homography
        bev_points = []
        for (u,v) in corners:
            uv1 = np.array([u,v,1.0])
            X = self.homography_matrix @ uv1
            if X[2] != 0:
                X /= X[2]
            bev_points.append((X[0], X[1]))

        # Convert BEV coordinates to occupancy grid indices and mark them
        self.mark_detection_in_grid(bev_points)

    def mark_detection_in_grid(self, bev_points):
        # Find bounding box in BEV coords (min_x, max_x, min_y, max_y)
        bev_x_coords = [p[0] for p in bev_points]
        bev_y_coords = [p[1] for p in bev_points]
        
        min_x = min(bev_x_coords)
        max_x = max(bev_x_coords)
        min_y = min(bev_y_coords)
        max_y = max(bev_y_coords)

        # Convert to grid coordinates
        # grid cell = (X - origin_x)/resolution
        def to_grid(x, y):
            gx = int((x - self.grid_origin_x) / self.grid_resolution)
            gy = int((y - self.grid_origin_y) / self.grid_resolution)
            return gx, gy

        gmin_x, gmin_y = to_grid(min_x, min_y)
        gmax_x, gmax_y = to_grid(max_x, max_y)

        # Clamp values to grid boundaries
        gmin_x = max(0, min(gmin_x, self.grid_width - 1))
        gmax_x = max(0, min(gmax_x, self.grid_width - 1))
        gmin_y = max(0, min(gmin_y, self.grid_height - 1))
        gmax_y = max(0, min(gmax_y, self.grid_height - 1))

        # Mark these cells as occupied (e.g., 100)
        for gx in range(gmin_x, gmax_x + 1):
            for gy in range(gmin_y, gmax_y + 1):
                idx = gy * self.grid_width + gx
                self.occupancy_data[idx] = 100

        self.get_logger().info("Marked detection area on BEV occupancy grid.")

    def publish_occupancy_grid(self):
        # Create occupancy grid message
        occ_msg = OccupancyGrid()
        occ_msg.header = Header()
        occ_msg.header.stamp = self.get_clock().now().to_msg()
        occ_msg.header.frame_id = "map"  # or "base_link", adjust as needed

        occ_msg.info.map_load_time = occ_msg.header.stamp
        occ_msg.info.resolution = self.grid_resolution
        occ_msg.info.width = self.grid_width
        occ_msg.info.height = self.grid_height
        occ_msg.info.origin.position.x = self.grid_origin_x
        occ_msg.info.origin.position.y = self.grid_origin_y
        occ_msg.info.origin.position.z = 0.0
        occ_msg.info.origin.orientation.w = 1.0

        occ_msg.data = self.occupancy_data
        self.occ_pub.publish(occ_msg)
        self.get_logger().debug("Published occupancy grid.")

def main(args=None):
    rclpy.init(args=args)
    node = BEVOccupancyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
