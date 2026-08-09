# import asyncio
# import time
# import rclpy
# from rclpy.action import ActionServer
# from rclpy.node import Node
# # from nav2_msgs.action import NavigateToPose
# from geometry_msgs.msg import PoseStamped
# from kobuki_interfaces.action import RobotNav

# class MinimalActionServer(Node):
#     def __init__(self):
#         super().__init__('nav_action_server')
#         self._action_server = ActionServer(
#             self,
#             RobotNav,
#             '/robot_nav',
#             self.execute_callback)
#         self.get_logger().info("Action Server Started. Waiting for Goal from UI...")

#     async def execute_callback(self, goal_handle):
#         self.get_logger().info('Executing goal...')
        
#         # Get the target coordinates from the UI request
#         target_x = goal_handle.request.pose.pose.position.x
#         target_y = goal_handle.request.pose.pose.position.y
        
#         feedback_msg = RobotNav.Feedback()
        
#         # Simulating movement (from 10 meters away down to 0)
#         for i in range(10, 0, -1):
#             feedback_msg.distance_remaining = float(i)
#             self.get_logger().info(f'Feedback: {i}m remaining to ({target_x}, {target_y})')
            
#             # Send feedback back to React UI
#             goal_handle.publish_feedback(feedback_msg)
#             await asyncio.sleep(1) # Simulate 1 second of driving

#         goal_handle.succeed()
#         result = RobotNav.Result()
#         return result

# def main(args=None):
#     rclpy.init(args=args)
#     minimal_action_server = MinimalActionServer()
#     rclpy.spin(minimal_action_server)

# if __name__ == '__main__':
#     main()
import rclpy
import time
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from kobuki_interfaces.action import RobotNav

class KobukiNavServer(Node):
    def __init__(self):
        super().__init__('nav_action_server')
        self._action_server = ActionServer(
            self,
            RobotNav,
            'robot_nav',
            self.execute_callback
        )
        self.get_logger().info('Action Server "robot_nav" is started...')

    def execute_callback(self, goal_handle):
        self.get_logger().info('Goal received! Moving to coordinates...')
        
        # Extract the goal coordinates
        target_x = goal_handle.request.pose.pose.position.x
        target_y = goal_handle.request.pose.pose.position.y
        
        feedback_msg = RobotNav.Feedback()
        
        # Simulate movement with feedback
        for i in range(5, 0, -1):
            feedback_msg.distance_remaining = float(i)
            self.get_logger().info(f'Distance remaining: {i}m')
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1.0) # Simulate travel time

        goal_handle.succeed()
        
        result = RobotNav.Result()
        result.success = True
        self.get_logger().info('Goal reached successfully!')
        return result

def main(args=None):
    rclpy.init(args=args)
    node = KobukiNavServer()
    # Critical: Actions need the MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()