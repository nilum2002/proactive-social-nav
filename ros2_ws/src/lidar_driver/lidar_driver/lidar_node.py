import math
import threading

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan

from lidar_driver.lidar_driver import LIDAR


class LidarNode(Node):
    """
    Publishes one LaserScan per lidar revolution.

    The LD-series sensor streams 12-point packets covering ~8 degrees each.
    Publishing a packet directly gives slam_toolbox an 8 degree sliver instead
    of a 360 degree sweep, so points are binned by angle here and released only
    when the sensor wraps past 0 degrees.

    Reading runs in its own thread: the sensor emits packets far faster than a
    10 Hz timer can drain them, and a timer-driven read leaves the serial buffer
    backing up so the data lags further behind real time the longer it runs.
    """

    def __init__(self):
        super().__init__('lidar_node')

        self.declare_parameter('serial_port', '/dev/ldlidar')
        self.declare_parameter('baudrate', 230400)
        self.declare_parameter('frame_id', 'laser')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('range_min', 0.05)
        self.declare_parameter('range_max', 8.0)
        # One bin per ~0.8 deg: the sensor yields roughly 450 points per turn at
        # 10 Hz.  Raising this past the real point count leaves permanent gaps.
        self.declare_parameter('angle_bins', 450)
        # LD-series report angle increasing clockwise; REP-103 wants yaw
        # counter-clockwise.  If the map comes out mirrored, flip this.
        self.declare_parameter('invert_angle', True)
        self.declare_parameter('angle_offset_deg', 0.0)

        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.range_min = float(self.get_parameter('range_min').value)
        self.range_max = float(self.get_parameter('range_max').value)
        self.bins = max(12, int(self.get_parameter('angle_bins').value))
        self.invert_angle = bool(self.get_parameter('invert_angle').value)
        self.angle_offset = float(self.get_parameter('angle_offset_deg').value)

        self.lidar = LIDAR(serial_port=serial_port, baudrate=baudrate)
        self.publisher = self.create_publisher(LaserScan, '/scan', 10)

        self._lock = threading.Lock()
        self._ranges = [math.inf] * self.bins
        self._intensities = [0.0] * self.bins
        self._last_start_angle = None
        self._sweep_started_at = self.get_clock().now()
        self._points_this_sweep = 0

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'LiDAR node started on {serial_port} at {baudrate} baud, '
            f'{self.bins} bins per revolution'
        )

    def _read_loop(self):
        while self._running:
            try:
                data = self.lidar.read_lidar_data()
            except Exception as exc:                      # serial hiccup
                self.get_logger().warn(f'LiDAR read failed: {exc}', throttle_duration_sec=5.0)
                continue

            if not data or not data.get('scan_data'):
                continue

            with self._lock:
                for point in data['scan_data']:
                    distance = point['distance'] / 1000.0
                    if distance <= 0.0 or distance < self.range_min or distance > self.range_max:
                        continue

                    angle = point['angle']
                    if self.invert_angle:
                        angle = 360.0 - angle
                    angle = (angle + self.angle_offset) % 360.0

                    idx = int(angle / 360.0 * self.bins) % self.bins
                    self._ranges[idx] = distance
                    self._intensities[idx] = float(point['intensity'])
                    self._points_this_sweep += 1

                # A large backwards jump in the raw start angle means the head
                # has passed 0 degrees, so the sweep in the buffer is complete.
                start_angle = data['start_angle']
                wrapped = (
                    self._last_start_angle is not None
                    and start_angle < self._last_start_angle - 180.0
                )
                self._last_start_angle = start_angle

                if wrapped:
                    self._publish_sweep()

    def _publish_sweep(self):
        """Emit the accumulated revolution and start a fresh one. Caller holds the lock."""
        now = self.get_clock().now()
        scan_time = (now - self._sweep_started_at).nanoseconds * 1e-9
        if scan_time <= 0.0:
            scan_time = 0.1

        scan_msg = LaserScan()
        scan_msg.header.stamp = self._sweep_started_at.to_msg()
        scan_msg.header.frame_id = self.frame_id
        scan_msg.angle_min = 0.0
        scan_msg.angle_increment = 2.0 * math.pi / self.bins
        scan_msg.angle_max = 2.0 * math.pi - scan_msg.angle_increment
        scan_msg.scan_time = scan_time
        scan_msg.time_increment = scan_time / self.bins
        scan_msg.range_min = self.range_min
        scan_msg.range_max = self.range_max
        scan_msg.ranges = list(self._ranges)
        scan_msg.intensities = list(self._intensities)

        self.publisher.publish(scan_msg)

        if self._points_this_sweep < self.bins // 4:
            self.get_logger().warn(
                f'Only {self._points_this_sweep} points in this revolution '
                f'({self.bins} bins) - check the sensor is spinning freely',
                throttle_duration_sec=10.0,
            )

        self._ranges = [math.inf] * self.bins
        self._intensities = [0.0] * self.bins
        self._points_this_sweep = 0
        self._sweep_started_at = now

    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
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
