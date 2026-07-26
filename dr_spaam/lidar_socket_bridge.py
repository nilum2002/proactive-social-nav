#!/usr/bin/env python3
"""gRPC LiDAR Bridge Node (Raspberry Pi 4)

Subscribes to ROS 2 topic (/scan) and streams binary Protobuf LaserScan
messages over gRPC (port 50051) to Jetson Nano for DR-SPAAM inference.
"""

import sys
import os
import time
import queue
import threading
from concurrent import futures
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import grpc

# Add local path to import generated protobuf modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lidar_stream_pb2
import lidar_stream_pb2_grpc

GRPC_PORT = 50051


class LidarServiceServicer(lidar_stream_pb2_grpc.LidarServiceServicer):
    """gRPC Service implementation that streams LiDAR data."""

    def __init__(self, scan_queue: queue.Queue, logger):
        self.scan_queue = scan_queue
        self.logger = logger

    def StreamScan(self, request, context):
        self.logger.info(f"✅ Jetson Nano connected over gRPC! ({context.peer()})")
        try:
            while context.is_active():
                try:
                    # Get scan message from ROS2 callback queue (timeout 1.0s)
                    scan_msg = self.scan_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Convert ROS2 LaserScan message to Protobuf LaserScanData
                pb_msg = lidar_stream_pb2.LaserScanData(
                    timestamp=time.time(),
                    angle_min=float(scan_msg.angle_min),
                    angle_max=float(scan_msg.angle_max),
                    angle_increment=float(scan_msg.angle_increment),
                    range_min=float(scan_msg.range_min),
                    range_max=float(scan_msg.range_max),
                    ranges=[
                        float(r) if not (np.isnan(r) or np.isinf(r)) else 29.99
                        for r in scan_msg.ranges
                    ],
                )
                yield pb_msg

        except Exception as e:
            self.logger.warn(f"gRPC stream ended: {e}")
        finally:
            self.logger.info("Jetson Nano gRPC connection closed.")


class LidarGrpcBridgeNode(Node):
    """ROS 2 Node that receives /scan and forwards to gRPC server queue."""

    def __init__(self):
        super().__init__("lidar_grpc_bridge")

        self.declare_parameter("scan_topic", "/scan")
        self.scan_topic = self.get_parameter("scan_topic").value

        # Thread-safe queue for buffering scan messages
        self.scan_queue = queue.Queue(maxsize=10)

        # ── Start gRPC Server ───────────────────────────────────────────────
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        self.servicer = LidarServiceServicer(self.scan_queue, self.get_logger())
        lidar_stream_pb2_grpc.add_LidarServiceServicer_to_server(
            self.servicer, self.grpc_server
        )

        server_address = f"0.0.0.0:{GRPC_PORT}"
        self.grpc_server.add_insecure_port(server_address)
        self.grpc_server.start()

        self.get_logger().info(
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║      gRPC LiDAR Stream Bridge Server     ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
            f"  gRPC Listening on: {server_address}\n"
            f"  Subscribed to ROS2 topic: {self.scan_topic}\n"
            f"  Waiting for Jetson Nano gRPC client..."
        )

        # Subscribe to ROS 2 /scan topic
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)

    def _scan_callback(self, msg: LaserScan):
        # Drop oldest scan if queue is full to prevent latency buildup
        if self.scan_queue.full():
            try:
                self.scan_queue.get_nowait()
            except queue.Empty:
                pass
        self.scan_queue.put(msg)

    def destroy_node(self):
        self.grpc_server.stop(grace=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarGrpcBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()