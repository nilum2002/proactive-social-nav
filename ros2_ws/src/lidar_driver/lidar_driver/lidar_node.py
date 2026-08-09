import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan

from lidar_driver.lidar_driver import LIDAR


class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 230400)
        self.declare_parameter('frame_id', 'laser')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('range_min', 0.05)
        self.declare_parameter('range_max', 8.0)

        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.range_min = float(self.get_parameter('range_min').value)
        self.range_max = float(self.get_parameter('range_max').value)

        self.lidar = LIDAR(serial_port=serial_port, baudrate=baudrate)
        self.publisher = self.create_publisher(LaserScan, '/scan', 10)
        self.timer = self.create_timer(1.0 / max(publish_rate, 0.1), self.publish_scan)

        self.get_logger().info(f'LiDAR node started on {serial_port} at {baudrate} baud')

    def publish_scan(self):
        data = self.lidar.read_lidar_data()
        if not data:
            return

        scan_data = data['scan_data']
        if not scan_data:
            return

        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = self.frame_id

        angles_rad = [math.radians(point['angle']) for point in scan_data]
        distances_m = [point['distance'] / 1000.0 for point in scan_data]
        intensities = [float(point['intensity']) for point in scan_data]

        scan_msg.angle_min = min(angles_rad)
        scan_msg.angle_max = max(angles_rad)
        if len(angles_rad) > 1:
            scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / (len(angles_rad) - 1)
        else:
            scan_msg.angle_increment = 0.0

        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / max(self.get_parameter('publish_rate').value, 0.1)
        scan_msg.range_min = self.range_min
        scan_msg.range_max = self.range_max
        scan_msg.ranges = [
            max(self.range_min, min(distance, self.range_max)) if distance > 0 else float('inf')
            for distance in distances_m
        ]
        scan_msg.intensities = intensities

        self.publisher.publish(scan_msg)

    def destroy_node(self):
        try:
            self.lidar.close_serial_connection()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
