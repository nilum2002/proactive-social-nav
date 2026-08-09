#!/usr/bin/env python3
"""
nav2_coordinator_node.py

Exposes a 'navigate_to_coordinate' service that tries two strategies:

  1. Nav2 (NavigateToPose action) — full path planning + obstacle avoidance.
     Requires a live /map and Nav2 stack.  Attempted first.

  2. Direct straight-line drive via /cmd_vel — no map, no obstacle avoidance.
     Used when Nav2 is unavailable OR when the caller sets use_direct_drive=true.

Service field  use_direct_drive (bool, default false):
  false  →  try Nav2 first; fall back to direct drive if Nav2 is unavailable.
  true   →  skip Nav2 entirely and use direct drive immediately.

The node reads the robot's current position from /odom (geometry_msgs/Odometry)
for the straight-line controller.  It does NOT use /map for that mode.
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from nav_coordinator_interfaces.srv import NavigateToCoordinate


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quaternion_to_yaw(q) -> float:
    """Extract yaw (rotation around Z) from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference a - b, result in (-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def wait_for_future(future, timeout: float | None = None):
    """
    Block until `future` resolves, without spinning the node.

    rclpy.spin_until_future_complete() cannot be used here: this runs inside a
    service callback while main()'s MultiThreadedExecutor already owns the node,
    and spinning it from a second executor deadlocks.  The executor thread is
    what completes the future, so we only have to wait on it.
    """
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        return None
    return future.result()


# ──────────────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────────────

class Nav2CoordinatorNode(Node):

    # ── tunables (could be ROS parameters) ────────────────────────────────────
    LINEAR_SPEED   = 0.25   # m/s  – straight-line mode
    ANGULAR_SPEED  = 0.5    # rad/s – straight-line mode
    CONTROL_HZ     = 20.0   # Hz   – straight-line control loop
    ODOM_TIMEOUT   = 3.0    # s    – how long to wait for first odom message
    GOAL_ACCEPT_TIMEOUT = 10.0  # s – how long to wait for Nav2 to accept a goal

    def __init__(self):
        super().__init__('nav2_coordinator_node')

        self._cb_group = ReentrantCallbackGroup()

        # ── Nav2 action client ────────────────────────────────────────────────
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose',
            callback_group=self._cb_group,
        )

        # ── cmd_vel publisher (direct drive) ─────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ── Odometry subscriber (direct drive needs current pose) ─────────────
        self._odom_lock = threading.Lock()
        self._odom: Odometry | None = None
        self._odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self._odom_callback,
            10,
            callback_group=self._cb_group,
        )

        # ── navigation service ────────────────────────────────────────────────
        self._service = self.create_service(
            NavigateToCoordinate,
            'navigate_to_coordinate',
            self._handle_navigate_request,
            callback_group=self._cb_group,
        )

        self._goal_lock = threading.Lock()
        self._active_goal_handle = None

        self.get_logger().info('Nav2CoordinatorNode is ready (Nav2 + direct-drive modes).')

    # ── Odometry ──────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        with self._odom_lock:
            self._odom = msg

    def _get_odom(self) -> Odometry | None:
        with self._odom_lock:
            return self._odom

    # ── Service handler ───────────────────────────────────────────────────────

    def _handle_navigate_request(self, request, response):
        self.get_logger().info(
            f'Received navigation request: x={request.x:.3f}, y={request.y:.3f}, '
            f'theta={request.theta:.3f}, xy_tol={request.xy_tolerance:.3f}, '
            f'yaw_tol={request.yaw_tolerance:.3f}'
        )

        # Caller can force direct drive via use_direct_drive field.
        # Fall back gracefully if the field doesn't exist in the .srv yet.
        use_direct = getattr(request, 'use_direct_drive', False)

        if use_direct:
            self.get_logger().info('use_direct_drive=true → skipping Nav2.')
            return self._drive_direct(request, response)

        # Try Nav2 first.
        nav2_available = self._action_client.wait_for_server(timeout_sec=3.0)

        if not nav2_available:
            self.get_logger().warn(
                'Nav2 action server not available — falling back to direct drive.'
            )
            return self._drive_direct(request, response)

        return self._navigate_nav2(request, response)

    # ── Strategy 1: Nav2 ──────────────────────────────────────────────────────

    def _navigate_nav2(self, request, response):
        """Send goal to Nav2 NavigateToPose and block until done."""

        goal_msg = NavigateToPose.Goal()

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = request.x
        pose.pose.position.y = request.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = yaw_to_quaternion(request.theta)
        goal_msg.pose = pose

        if hasattr(goal_msg, 'behavior_tree'):
            goal_msg.behavior_tree = ''

        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback,
        )
        goal_handle = wait_for_future(send_future, timeout=self.GOAL_ACCEPT_TIMEOUT)

        if goal_handle is None or not goal_handle.accepted:
            msg = 'Nav2 rejected the goal — falling back to direct drive.'
            self.get_logger().warn(msg)
            return self._drive_direct(request, response)

        self.get_logger().info('Goal accepted by Nav2, waiting for result…')

        with self._goal_lock:
            self._active_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        wrapped = wait_for_future(result_future)

        with self._goal_lock:
            self._active_goal_handle = None

        if wrapped is None:
            msg = 'No result from Nav2 — falling back to direct drive.'
            self.get_logger().error(msg)
            return self._drive_direct(request, response)

        status = wrapped.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            response.success = True
            response.message = 'Navigation succeeded (Nav2).'
            self.get_logger().info(response.message)

        elif status == GoalStatus.STATUS_CANCELED:
            response.success = False
            response.message = 'Navigation cancelled (Nav2).'
            self.get_logger().warn(response.message)

        elif status == GoalStatus.STATUS_ABORTED:
            response.success = False
            response.message = 'Navigation aborted by Nav2 (obstacle or planner failure).'
            self.get_logger().error(response.message)

        else:
            response.success = False
            response.message = f'Unexpected Nav2 status code: {status}.'
            self.get_logger().error(response.message)

        return response

    def _feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        remaining = getattr(fb, 'distance_remaining', float('nan'))

        nav_time = float('nan')
        if hasattr(fb, 'navigation_time'):
            nav_time = fb.navigation_time.sec + fb.navigation_time.nanosec * 1e-9

        eta = float('nan')
        if hasattr(fb, 'estimated_time_remaining'):
            eta = fb.estimated_time_remaining.sec + fb.estimated_time_remaining.nanosec * 1e-9

        self.get_logger().info(
            f'[Nav2 Feedback] dist={remaining:.2f} m  '
            f'elapsed={nav_time:.1f} s  eta={eta:.1f} s'
        )

    # ── Strategy 2: direct straight-line drive ────────────────────────────────

    def _drive_direct(self, request, response):
        """
        Simple open-loop straight-line controller.
        Phase 1 – rotate in place to face the target.
        Phase 2 – drive straight toward the target.
        Phase 3 – rotate in place to reach the desired final heading (theta).

        Uses /odom for pose feedback.  No obstacle avoidance.
        """
        self.get_logger().info('Direct-drive mode: no map, no obstacle avoidance.')

        # Wait for the first odom message.
        deadline = self.get_clock().now().nanoseconds * 1e-9 + self.ODOM_TIMEOUT
        while self._get_odom() is None:
            time.sleep(0.05)
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                msg = 'Direct drive failed: no /odom received within timeout.'
                self.get_logger().error(msg)
                response.success = False
                response.message = msg
                return response

        dt = 1.0 / self.CONTROL_HZ
        tol_xy  = max(request.xy_tolerance,  0.05)   # floor at 5 cm
        tol_yaw = max(request.yaw_tolerance, 0.05)   # floor at ~3 °

        # ── Phase 1: rotate to face goal ─────────────────────────────────────
        self.get_logger().info('Direct drive phase 1: rotate to face goal.')
        while rclpy.ok():
            odom = self._get_odom()
            cx = odom.pose.pose.position.x
            cy = odom.pose.pose.position.y
            cyaw = quaternion_to_yaw(odom.pose.pose.orientation)

            dx = request.x - cx
            dy = request.y - cy
            dist = math.hypot(dx, dy)

            if dist < tol_xy:
                # Already at the goal position — skip straight to phase 3.
                break

            bearing = math.atan2(dy, dx)
            err = angle_diff(bearing, cyaw)

            if abs(err) < tol_yaw:
                break

            twist = Twist()
            twist.angular.z = math.copysign(self.ANGULAR_SPEED, err)
            self._cmd_vel_pub.publish(twist)
            time.sleep(dt)

        self._stop()

        # ── Phase 2: drive straight to goal ──────────────────────────────────
        self.get_logger().info('Direct drive phase 2: drive to goal.')
        while rclpy.ok():
            odom = self._get_odom()
            cx = odom.pose.pose.position.x
            cy = odom.pose.pose.position.y

            dx = request.x - cx
            dy = request.y - cy
            dist = math.hypot(dx, dy)

            if dist < tol_xy:
                break

            # Small heading correction while driving.
            cyaw = quaternion_to_yaw(odom.pose.pose.orientation)
            bearing = math.atan2(dy, dx)
            heading_err = angle_diff(bearing, cyaw)

            twist = Twist()
            twist.linear.x  = self.LINEAR_SPEED
            twist.angular.z = math.copysign(
                min(self.ANGULAR_SPEED, abs(heading_err) * 2.0),
                heading_err,
            )
            self._cmd_vel_pub.publish(twist)
            time.sleep(dt)

        self._stop()

        # ── Phase 3: rotate to final heading (theta) ──────────────────────────
        self.get_logger().info('Direct drive phase 3: rotate to final heading.')
        while rclpy.ok():
            odom = self._get_odom()
            cyaw = quaternion_to_yaw(odom.pose.pose.orientation)
            err = angle_diff(request.theta, cyaw)

            if abs(err) < tol_yaw:
                break

            twist = Twist()
            twist.angular.z = math.copysign(self.ANGULAR_SPEED, err)
            self._cmd_vel_pub.publish(twist)
            time.sleep(dt)

        self._stop()

        response.success = True
        response.message = 'Navigation succeeded (direct drive, no obstacle avoidance).'
        self.get_logger().info(response.message)
        return response

    def _stop(self):
        """Publish a zero-velocity command."""
        self._cmd_vel_pub.publish(Twist())


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    node = Nav2CoordinatorNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Nav2CoordinatorNode.')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()