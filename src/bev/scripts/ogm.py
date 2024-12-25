#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# ROS messages
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, TransformStamped
from std_msgs.msg import Header

# tf2 for transforms
import tf2_ros
import tf2_geometry_msgs  # Needed for do_transform_pose, etc.

# YOLO detection messages
from yolo_msgs.msg import Detection, DetectionArray

import numpy as np

# For polygon operations
try:
    from shapely.geometry import Polygon, Point
except ImportError:
    # If Shapely is not available, you need to install it:
    #   pip install shapely
    # or use your distro’s package manager
    raise RuntimeError("Shapely is required for polygon fill. Please install it.")


class GlobalOccupancyNode(Node):
    def __init__(self):
        super().__init__('global_occupancy_node')
        self.get_logger().info("Starting GlobalOccupancyNode")

        # Subscriber to bounding box detections
        self.detection_sub = self.create_subscription(
            DetectionArray,
            '/yolo/detections_3d',
            self.detections_callback,
            10
        )

        # Publisher for OccupancyGrid
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid,
            '/bev/global_occupancy',
            10
        )

        # --- TF Buffer & Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Define GLOBAL occupancy grid parameters ---
        # 1) Use "world" frame
        self.map_frame_id = "world"

        # 2) Start with a modest grid size, expand dynamically as needed
        self.resolution = 0.1
        self.grid_width = 200
        self.grid_height = 200
        self.origin_x = -10.0
        self.origin_y = -10.0

        # We keep a single global occupancy grid in memory
        self.global_occupancy_grid = self.create_empty_grid()

        # Identity homography (placeholder)
        self.H = np.eye(3)

    def detections_callback(self, msg: DetectionArray):
        """
        Callback for each DetectionArray message.
        Transform each bounding box to the 'world' frame, project, and fill occupancy.
        """
        if not msg.detections:
            return

        for detection in msg.detections:
            center_pose = detection.bbox3d.center  # geometry_msgs/Pose
            size = detection.bbox3d.size           # geometry_msgs/Vector3
            bbox_frame_id = detection.bbox3d.frame_id

            # Attempt transform from detection frame to world
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.map_frame_id,
                    bbox_frame_id,
                    rclpy.time.Time()
                )
            except tf2_ros.LookupException as e:
                self.get_logger().warn(
                    f"Could not find transform {bbox_frame_id} -> {self.map_frame_id}: {str(e)}"
                )
                continue

            # Compute 8 corners in local detection frame
            corners_3d = self.compute_3d_corners(center_pose, size)

            # Transform each corner to world frame
            corners_3d_world = []
            for corner in corners_3d:
                corner_pose = self.point_to_pose(corner)
                corner_pose_world = tf2_geometry_msgs.do_transform_pose(
                    corner_pose,
                    transform_stamped
                )
                corners_3d_world.append([
                    corner_pose_world.position.x,
                    corner_pose_world.position.y,
                    corner_pose_world.position.z
                ])

            # Project onto ground plane (2D)
            corners_2d = self.project_corners_homography(corners_3d_world)

            # Fill the global occupancy grid (entire area under the bounding box)
            self.fill_occupancy_grid(self.global_occupancy_grid, corners_2d)

        # Publish after processing all detections
        self.publish_global_occupancy()

    def point_to_pose(self, xyz):
        """
        Simple helper: convert (x,y,z) -> geometry_msgs/Pose
        """
        p = Pose()
        p.position.x = xyz[0]
        p.position.y = xyz[1]
        p.position.z = xyz[2]
        p.orientation.w = 1.0
        return p

    def compute_3d_corners(self, center_pose: Pose, size: Vector3):
        """
        Compute the 8 corners of the 3D bounding box relative to its center.
        No orientation applied. If needed, handle quaternion rotation here.
        """
        cx, cy, cz = center_pose.position.x, center_pose.position.y, center_pose.position.z
        dx, dy, dz = size.x, size.y, size.z

        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

        corners = np.array([
            [cx - hx, cy - hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz + hz],
        ])
        return corners

    def project_corners_homography(self, corners_3d_world):
        """
        Project 3D corners to 2D ground plane.
        For many use cases, just using (X, Y) is sufficient if Z ~ 0.
        Here, we use an identity homography as an example.
        """
        corners_2d = []
        for corner in corners_3d_world:
            X, Y, Z = corner[0], corner[1], corner[2]
            vec_3x1 = np.array([X, Y, 1.0])
            uvw = self.H @ vec_3x1
            u, v, w = uvw
            if abs(w) < 1e-9:
                w = 1e-9
            corners_2d.append([u / w, v / w])
        return corners_2d

    def create_empty_grid(self):
        """
        Create a nav_msgs/OccupancyGrid with the correct metadata,
        initially filled with -1 (unknown).
        """
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.map_frame_id

        grid.info.resolution = self.resolution
        grid.info.width = self.grid_width
        grid.info.height = self.grid_height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        num_cells = self.grid_width * self.grid_height
        grid.data = [-1] * num_cells
        return grid

    def expand_grid_if_needed(self, grid: OccupancyGrid, min_x, max_x, min_y, max_y):
        """
        Expand occupancy grid if the bounding box of the polygon
        lies outside the current grid. 
        We do the minimal expansion possible in each direction.
        """
        current_min_x = grid.info.origin.position.x
        current_min_y = grid.info.origin.position.y
        current_max_x = current_min_x + grid.info.width * grid.info.resolution
        current_max_y = current_min_y + grid.info.height * grid.info.resolution

        expand_left = 0
        expand_right = 0
        expand_down = 0
        expand_up = 0

        # Check how far we need to expand in each direction
        if min_x < current_min_x:
            expand_left = int(np.ceil((current_min_x - min_x) / grid.info.resolution))
        if max_x > current_max_x:
            expand_right = int(np.ceil((max_x - current_max_x) / grid.info.resolution))
        if min_y < current_min_y:
            expand_down = int(np.ceil((current_min_y - min_y) / grid.info.resolution))
        if max_y > current_max_y:
            expand_up = int(np.ceil((max_y - current_max_y) / grid.info.resolution))

        if (expand_left == 0 and expand_right == 0 and 
            expand_down == 0 and expand_up == 0):
            return  # No expansion needed

        # Save old grid metadata
        old_width = grid.info.width
        old_height = grid.info.height
        old_origin_x = grid.info.origin.position.x
        old_origin_y = grid.info.origin.position.y
        old_data = grid.data

        # Compute new origin (shifts if we expand left/down)
        new_origin_x = old_origin_x - expand_left * grid.info.resolution
        new_origin_y = old_origin_y - expand_down * grid.info.resolution

        new_width = old_width + expand_left + expand_right
        new_height = old_height + expand_down + expand_up

        # Create new data array, fill with -1
        new_data = [-1] * (new_width * new_height)

        # Copy the old data into the correct offset
        for row in range(old_height):
            for col in range(old_width):
                old_index = row * old_width + col
                val = old_data[old_index]

                new_row = row + expand_down
                new_col = col + expand_left
                new_index = new_row * new_width + new_col
                new_data[new_index] = val

        # Update grid metadata
        grid.info.width = new_width
        grid.info.height = new_height
        grid.info.origin.position.x = new_origin_x
        grid.info.origin.position.y = new_origin_y
        grid.data = new_data

    def fill_occupancy_grid(self, grid: OccupancyGrid, corners_2d):
        """
        1) Expand grid if needed to fit the bounding box of the projected polygon.
        2) Fill all cells whose center lies within the 2D polygon.
        """
        # 1) Figure out bounding box of corners in world coords
        min_x = min(pt[0] for pt in corners_2d)
        max_x = max(pt[0] for pt in corners_2d)
        min_y = min(pt[1] for pt in corners_2d)
        max_y = max(pt[1] for pt in corners_2d)

        # 2) Expand grid if needed
        self.expand_grid_if_needed(grid, min_x, max_x, min_y, max_y)

        # 3) Construct a Shapely polygon
        # Make sure corners_2d are in a sensible (clockwise or counterclockwise) order.
        # If they're not, you can sort them or reorder as needed. 
        polygon = Polygon(corners_2d)

        # 4) Convert bounding box to grid coords
        def world_to_grid(x, y):
            col = int((x - grid.info.origin.position.x) / grid.info.resolution)
            row = int((y - grid.info.origin.position.y) / grid.info.resolution)
            return row, col

        # Half a cell to check cell center
        half_res = grid.info.resolution / 2.0

        # Convert bounding box corners to grid indices (clamped within grid)
        grid_min_col, grid_min_row = world_to_grid(min_x, min_y)
        grid_max_col, grid_max_row = world_to_grid(max_x, max_y)

        # Clamp
        grid_min_col = max(0, grid_min_col)
        grid_min_row = max(0, grid_min_row)
        grid_max_col = min(grid.info.width - 1, grid_max_col)
        grid_max_row = min(grid.info.height - 1, grid_max_row)

        # 5) Iterate over all cells in bounding rectangle
        for row in range(grid_min_row, grid_max_row + 1):
            for col in range(grid_min_col, grid_max_col + 1):
                # Compute cell center in world coords
                cell_center_x = grid.info.origin.position.x + (col + 0.5) * grid.info.resolution
                cell_center_y = grid.info.origin.position.y + (row + 0.5) * grid.info.resolution

                # Check if center is inside the polygon
                if polygon.contains(Point(cell_center_x, cell_center_y)):
                    index = row * grid.info.width + col
                    grid.data[index] = 100  # Mark as occupied

    def publish_global_occupancy(self):
        """
        Publish the global occupancy grid. Update the timestamp each time.
        """
        self.global_occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        self.occupancy_pub.publish(self.global_occupancy_grid)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalOccupancyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
