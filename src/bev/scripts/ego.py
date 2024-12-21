#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np


class GPSConverter:
    def __init__(self):
        self.a = 6378137.0  # Semi-major axis
        self.f_inv = 298.257223563  # Inverse flattening
        self.f = 1.0 / self.f_inv

    def geodetic_to_ecef(self, lat, lon, alt):
        """
        Convert geodetic coordinates to ECEF coordinates.
        lat, lon in degrees, alt in meters.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        N = self.a / np.sqrt(1 - (self.f * (2 - self.f) * np.sin(lat_rad) ** 2))
        X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = ((1 - self.f) * (1 - self.f) * N + alt) * np.sin(lat_rad)

        return np.array([X, Y, Z])

    def geo_to_enu(self, geodetic, origin_lat, origin_lon, origin_alt):
        """
        Convert geodetic coordinates to ENU coordinates.
        geodetic: np.array([lat, lon, alt])
        origin_lat, origin_lon, origin_alt: reference point for ENU origin
        """
        # Convert origin to ECEF
        origin_ecef = self.geodetic_to_ecef(origin_lat, origin_lon, origin_alt)

        # Convert geodetic point to ECEF
        ecef = self.geodetic_to_ecef(geodetic[0], geodetic[1], geodetic[2])

        # Compute rotation matrix R
        lon_rad = np.radians(origin_lon)
        lat_rad = np.radians(origin_lat)

        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)

        R = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])

        # ENU coordinates
        enu = R @ (ecef - origin_ecef)
        return enu


class GPSOdometryNode(Node):
    def __init__(self):
        super().__init__('gps_to_odometry_node')
        self.get_logger().info("Starting GPS to ENU Odometry Node")

        # Subscriber to NavSatFix
        self.gps_subscriber = self.create_subscription(
            NavSatFix,
            '/kitti/oxts/gps/fix',
            self.gps_callback,
            10
        )

        # Publisher for Odometry
        self.odom_publisher = self.create_publisher(Odometry, '/ego/odometry', 10)

        # GPS Converter and origin setup
        self.converter = GPSConverter()
        self.origin_set = False
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None

    def gps_callback(self, msg: NavSatFix):
        """
        Callback function for GPS messages.
        """
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude

        # Set origin on the first GPS message
        if not self.origin_set:
            self.origin_lat = latitude
            self.origin_lon = longitude
            self.origin_alt = altitude
            self.origin_set = True
            self.get_logger().info(
                f"Origin set at Lat: {latitude}, Lon: {longitude}, Alt: {altitude}"
            )
            return

        # Convert GPS to ENU
        enu_coords = self.converter.geo_to_enu(
            [latitude, longitude, altitude],
            self.origin_lat,
            self.origin_lon,
            self.origin_alt
        )

        # Create and populate the Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Set ENU position
        odom_msg.pose.pose.position = Point(
            x=enu_coords[0],
            y=enu_coords[1],
            z=enu_coords[2]
        )

        # Orientation set to neutral (no rotation, ENU aligned)
        odom_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Publish the Odometry message
        self.odom_publisher.publish(odom_msg)
        self.get_logger().info(f"Published ENU Odometry: x={enu_coords[0]}, y={enu_coords[1]}, z={enu_coords[2]}")


def main(args=None):
    rclpy.init(args=args)
    node = GPSOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
