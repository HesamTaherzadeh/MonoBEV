#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# ROS Messages
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped

# TF
from tf2_ros import TransformBroadcaster

# Numpy
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

        # Flattening
        e2 = self.f * (2 - self.f)  # e^2 = 2f - f^2
        N = self.a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

        X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = ((1 - e2) * N + alt) * np.sin(lat_rad)

        return np.array([X, Y, Z])

    def geo_to_enu(self, geodetic, origin_lat, origin_lon, origin_alt):
        """
        Convert geodetic [lat, lon, alt] to ENU coordinates relative to
        the given origin (origin_lat, origin_lon, origin_alt).
        """
        # 1) Convert origin to ECEF
        origin_ecef = self.geodetic_to_ecef(origin_lat, origin_lon, origin_alt)

        # 2) Convert target geodetic to ECEF
        ecef = self.geodetic_to_ecef(geodetic[0], geodetic[1], geodetic[2])

        # 3) Compute ENU rotation matrix
        lon_rad = np.radians(origin_lon)
        lat_rad = np.radians(origin_lat)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)

        # ENU transform matrix
        # (east, north, up)
        R = np.array([
            [-sin_lon,          cos_lon,            0      ],
            [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat ],
            [ cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat ]
        ])

        # 4) Translate and rotate
        enu = R @ (ecef - origin_ecef)
        return enu


class GPSIMUOdometryNode(Node):
    def __init__(self):
        super().__init__('gps_imu_odometry_node')
        self.get_logger().info("Starting GPS + IMU to ENU Odometry Node")

        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self.odom_publisher = self.create_publisher(Odometry, '/ego/odometry', 10)

        # Subscribers
        self.gps_subscriber = self.create_subscription(
            NavSatFix,
            '/kitti/oxts/gps/fix',
            self.gps_callback,
            10
        )
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/kitti/oxts/imu',
            self.imu_callback,
            10
        )

        # GPS Converter
        self.converter = GPSConverter()

        # ENU origin (set once on the first GPS)
        self.origin_set = False
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None

        # Store the latest IMU orientation
        self.latest_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.have_imu_data = False

    def imu_callback(self, msg: Imu):
        """
        IMU callback - store the latest orientation in a member variable.
        """
        print("recieved IMU")
        self.latest_orientation = msg.orientation
        self.have_imu_data = True

    def gps_callback(self, msg: NavSatFix):
        """
        GPS callback - convert to ENU position, combine with IMU orientation, publish Odometry.
        """
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude

        # 1) If we haven't set the origin, use the first GPS as origin.
        if not self.origin_set:
            self.origin_lat = latitude
            self.origin_lon = longitude
            self.origin_alt = altitude
            self.origin_set = True

            self.get_logger().info(
                f"Origin set to lat={latitude}, lon={longitude}, alt={altitude}"
            )
            return

        # 2) Convert from lat/lon/alt to ENU
        enu_coords = self.converter.geo_to_enu(
            [latitude, longitude, altitude],
            self.origin_lat,
            self.origin_lon,
            self.origin_alt
        )

        # 3) Build the Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        # Frame convention: "odom" as the fixed ENU frame, "base_link" for the vehicle
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Position from GPS->ENU
        odom_msg.pose.pose.position.x = enu_coords[0]
        odom_msg.pose.pose.position.y = enu_coords[1]
        odom_msg.pose.pose.position.z = enu_coords[2]

        # Orientation from IMU
        # If no IMU data received yet, we leave it as identity quaternion
        if self.have_imu_data:
            odom_msg.pose.pose.orientation = self.latest_orientation
        else:
            # Orientation fallback: identity
            odom_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # 4) Broadcast TF
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = odom_msg.header.stamp
        transform_stamped.header.frame_id = "odom"
        transform_stamped.child_frame_id = "base_link"

        transform_stamped.transform.translation.x = odom_msg.pose.pose.position.x
        transform_stamped.transform.translation.y = odom_msg.pose.pose.position.y
        transform_stamped.transform.translation.z = odom_msg.pose.pose.position.z
        transform_stamped.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(transform_stamped)

        # 5) Publish the odometry
        self.odom_publisher.publish(odom_msg)
        self.get_logger().info(
            f"Published ENU Odom: pos=({enu_coords[0]:.2f}, {enu_coords[1]:.2f}, {enu_coords[2]:.2f})"
            f" | orientation=({odom_msg.pose.pose.orientation.x:.3f}, "
            f"{odom_msg.pose.pose.orientation.y:.3f}, {odom_msg.pose.pose.orientation.z:.3f}, "
            f"{odom_msg.pose.pose.orientation.w:.3f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GPSIMUOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
