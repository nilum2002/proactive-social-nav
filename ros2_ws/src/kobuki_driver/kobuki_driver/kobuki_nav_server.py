import rclpy
import threading
import time
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from kobuki_interfaces.srv import RobotNav, CancelRobotNav
from kobuki_interfaces.msg import RobotNavFeedback

class KobukiNavServer(Node):
    def __init__(self):
        super().__init__('kobuki_nav_server')
        
        self._state_lock = threading.Lock()
        self._goal_active = False
        self._cancel_requested = False
        self._worker_thread = None

        # Service to receive goal
        self.srv = self.create_service(RobotNav, 'robot_nav', self.handle_nav_goal)

        # Service to cancel the active goal
        self.cancel_srv = self.create_service(
            CancelRobotNav, 'robot_nav/cancel', self.handle_cancel_goal)
        
        # Topic to publish feedback
        self.feedback_pub = self.create_publisher(RobotNavFeedback, '/robot_nav/feedback', 10)
        
        self.get_logger().info('Kobuki Navigation Service Ready on /robot_nav and /robot_nav/cancel')

    def handle_nav_goal(self, request, response):
        with self._state_lock:
            if self._goal_active:
                response.success = False
                response.message = 'Another goal is already active'
                return response

            self._goal_active = True
            self._cancel_requested = False

        self.get_logger().info(
            f'Goal received → X={request.pose.pose.position.x}, Y={request.pose.pose.position.y}')

        self._worker_thread = threading.Thread(
            target=self._run_goal,
            args=(request.pose.pose.position.x, request.pose.pose.position.y),
            daemon=True,
        )
        self._worker_thread.start()

        response.success = True
        response.message = 'Goal accepted and navigation started'
        return response

    def handle_cancel_goal(self, request, response):
        del request

        with self._state_lock:
            if not self._goal_active:
                response.success = False
                response.message = 'No active goal to cancel'
                return response

            self._cancel_requested = True

        feedback = RobotNavFeedback()
        feedback.distance_remaining = 0.0
        feedback.status = 'Cancel requested'
        self.feedback_pub.publish(feedback)

        response.success = True
        response.message = 'Cancel request accepted'
        self.get_logger().info('Cancel request accepted for active goal')
        return response

    def _run_goal(self, target_x, target_y):
        feedback = RobotNavFeedback()

        self.get_logger().info(f'Running navigation toward X={target_x}, Y={target_y}')

        for remaining in range(10, 0, -1):
            with self._state_lock:
                if self._cancel_requested:
                    feedback.distance_remaining = float(remaining)
                    feedback.status = 'Goal canceled'
                    self.feedback_pub.publish(feedback)
                    self.get_logger().info('Goal canceled by request')
                    self._goal_active = False
                    return

            feedback.distance_remaining = float(remaining)
            feedback.status = f'Moving... {remaining}m remaining'
            self.feedback_pub.publish(feedback)
            self.get_logger().info(f'Feedback: {remaining}m left')
            time.sleep(1.0)

        feedback.distance_remaining = 0.0
        feedback.status = 'Goal reached successfully'
        self.feedback_pub.publish(feedback)

        with self._state_lock:
            self._goal_active = False
            self._cancel_requested = False

        self.get_logger().info('Goal reached successfully!')

def main(args=None):
    rclpy.init(args=args)
    node = KobukiNavServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()