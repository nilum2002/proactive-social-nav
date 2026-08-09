import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from kobuki_driver.kuboki_driver import KobukiDriver


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


class KobukiRosNode(Node):
    def __init__(self):
        super().__init__('kobuki_node')

        # Parameters (can be set via ros2 param later)
        self.declare_parameter('port', '/dev/kobuki')
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('cmd_rate', 10)
        self.declare_parameter('odom_pub_rate', 10)

        port = self.get_parameter('port').get_parameter_value().string_value
        cmd_timeout = self.get_parameter('cmd_timeout').get_parameter_value().double_value
        cmd_rate = int(self.get_parameter('cmd_rate').get_parameter_value().integer_value)
        odom_rate = int(self.get_parameter('odom_pub_rate').get_parameter_value().integer_value)

        # Initialize driver
        try:
            self.driver = KobukiDriver(port=port, cmd_timeout=cmd_timeout, cmd_rate=cmd_rate)
        except Exception as e:
            self.get_logger().error(f'Failed to initialize KobukiDriver: {e}')
            raise

        # Subscribers and publishers
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)

        # Timer to publish odom and imu
        self._odom_timer = self.create_timer(1.0 / max(1, odom_rate), self._publish_state)

        self.get_logger().info('Kobuki ROS node initialized')

    def cmd_vel_cb(self, msg: Twist):
        # Convert linear m/s -> mm/s for driver, pass angular z as rad/s
        linear_m_s = msg.linear.x
        angular_rad_s = msg.angular.z
        linear_mm_s = linear_m_s * 1000.0
        # Use set_velocity which stores and lets driver send commands
        self.driver.set_velocity(int(linear_mm_s), angular_rad_s)

    def _publish_state(self):
        state = self.driver.get_state()
        now = self.get_clock().now().to_msg()

        # Odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = state.get('x', 0.0)
        odom.pose.pose.position.y = state.get('y', 0.0)
        odom.pose.pose.position.z = 0.0

        q = yaw_to_quaternion(state.get('theta', 0.0))
        odom.pose.pose.orientation = q

        # Twist in child frame
        odom.twist.twist.linear.x = state.get('linear_velocity', 0.0)
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = state.get('angular_velocity', 0.0)

        self.odom_pub.publish(odom)

        # IMU
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = 'base_link'

        # Put orientation from pose (yaw) into IMU orientation
        imu.orientation_covariance[0] = -1

        # Gyro: driver provides gyro_rate in deg/s; convert to rad/s
        gyro_deg_s = state.get('gyro_rate', 0.0)
        imu.angular_velocity.x = 0.0
        imu.angular_velocity.y = 0.0
        imu.angular_velocity.z = math.radians(gyro_deg_s)

        # No linear acceleration from driver
        imu.linear_acceleration.x = 0.0
        imu.linear_acceleration.y = 0.0
        imu.linear_acceleration.z = 0.0

        self.imu_pub.publish(imu)


def main(args=None):
    rclpy.init(args=args)
    node = KobukiRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Kobuki node')
        node.driver.set_velocity(0, 0)
        node.driver.running = False
        rclpy.shutdown()


if __name__ == '__main__':
    main()
