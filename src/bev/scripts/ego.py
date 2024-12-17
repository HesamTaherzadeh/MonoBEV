#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from pyproj import CRS, Transformer
import math


class EgoOdometryNode(Node):
    def __init__(self):
        super().__init__('ego_node')
        self.get_logger().info("Ego Node: Subscribing to IMU and GPS data with Approximate Sync")

        # Subscribers with message_filters
        self.imu_sub = Subscriber(self, Imu, '/kitti/oxts/imu')
        self.gps_sub = Subscriber(self, NavSatFix, '/kitti/oxts/gps/fix')
        self.vel_sub = Subscriber(self, TwistStamped, '/kitti/oxts/gps/vel')

        # Synchronize the topics with ApproximateTimeSynchronizer
        self.sync = ApproximateTimeSynchronizer([self.imu_sub, self.gps_sub, self.vel_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.sync_callback)

        # Publisher for Odometry
        self.odom_publisher = self.create_publisher(Odometry, '/ego/odometry', 10)

        # Flag and storage for origin
        self.origin_set = False
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None

        # Setup transformers
        self.wgs84 = CRS.from_epsg(4326)   # WGS84 lat/lon
        self.ecef = CRS.from_epsg(4978)    # ECEF
        self.transform_to_ecef = Transformer.from_crs(self.wgs84, self.ecef, always_xy=True)

    def sync_callback(self, imu_msg: Imu, gps_msg: NavSatFix, vel_msg: TwistStamped):
        """
        Callback function for synchronized IMU, GPS, and velocity data.
        """
        # Store the first GPS reading as origin
        if not self.origin_set:
            self.origin_lat = gps_msg.latitude
            self.origin_lon = gps_msg.longitude
            self.origin_alt = gps_msg.altitude
            self.origin_x, self.origin_y, self.origin_z = self.latlon_to_ecef(self.origin_lat, self.origin_lon, self.origin_alt)
            self.origin_set = True
            self.get_logger().info("Origin set at Lat: {:.6f}, Lon: {:.6f}, Alt: {:.2f}".format(
                self.origin_lat, self.origin_lon, self.origin_alt
            ))

        # Convert current GPS to ENU
        position_e, position_n, position_u = self.latlon_to_enu(gps_msg.latitude, gps_msg.longitude, gps_msg.altitude)

        # Extract orientation (from IMU)
        orientation_quat = imu_msg.orientation

        # Extract velocity
        velocity = vel_msg.twist.linear

        # Create and populate the Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = gps_msg.header.stamp  # Use GPS timestamp for consistency
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "world"

        # Position in ENU frame
        odom_msg.pose.pose.position.x = position_e
        odom_msg.pose.pose.position.y = position_n
        odom_msg.pose.pose.position.z = 0.0

        # Orientation
        odom_msg.pose.pose.orientation = orientation_quat

        # Publish Odometry
        self.odom_publisher.publish(odom_msg)
        self.get_logger().info(f"Published Odometry at time: {gps_msg.header.stamp.sec}.{gps_msg.header.stamp.nanosec}")

    def latlon_to_ecef(self, lat, lon, alt):
        """
        Convert lat/lon/alt (WGS84) to ECEF coordinates (X,Y,Z).
        lat, lon in degrees, alt in meters.
        """
        # Note: Transformer expects coordinates in (lon, lat, alt) and in degrees.
        x, y, z = self.transform_to_ecef.transform(lat, lon, alt)
        return x, y, z

    def latlon_to_enu(self, lat, lon, alt):
        """
        Convert lat/lon/alt to local ENU coordinates relative to the stored origin.
        """
        # Get ECEF of current point
        X, Y, Z = self.latlon_to_ecef(lat, lon, alt)

        # Differences from origin
        dX = X - self.origin_x
        dY = Y - self.origin_y
        dZ = Z - self.origin_z

        # Convert lat, lon to radians for trig
        lat0 = math.radians(self.origin_lat)
        lon0 = math.radians(self.origin_lon)

        # ECEF to ENU rotation
        # Reference: standard formulas for ECEF->ENU
        sin_lat0 = math.sin(lat0)
        cos_lat0 = math.cos(lat0)
        sin_lon0 = math.sin(lon0)
        cos_lon0 = math.cos(lon0)

        # Rotation matrix applied to delta ECEF coordinates:
        # [e]   [ -sin_lon0        cos_lon0           0 ] [dX]
        # [n] = [ -cos_lon0*sin_lat0 -sin_lon0*sin_lat0 cos_lat0 ] [dY]
        # [u]   [ cos_lon0*cos_lat0  sin_lon0*cos_lat0 sin_lat0  ] [dZ]

        e = -sin_lon0*dX + cos_lon0*dY
        n = (-cos_lon0*sin_lat0)*dX + (-sin_lon0*sin_lat0)*dY + cos_lat0*dZ
        u = (cos_lon0*cos_lat0)*dX + (sin_lon0*cos_lat0)*dY + sin_lat0*dZ

        return dX, dY, dZ


def main(args=None):
    rclpy.init(args=args)
    node = EgoOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
