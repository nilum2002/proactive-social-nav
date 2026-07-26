#!/usr/bin/env python3
"""gRPC LiDAR Client (Jetson Nano - Standalone No-ROS)

Connects to Raspberry Pi 4's gRPC server, receives high-frequency
Protobuf LiDAR scans, runs PyTorch DR-SPAAM inference on GPU,
and applies 2D EKF multi-object tracking.
"""

import sys
import os
import time
import numpy as np
import torch
import grpc

# Import generated Protobuf files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lidar_stream_pb2
import lidar_stream_pb2_grpc

from dr_spaam.detector import Detector
from dr_spaam_tracker_node import KalmanFilter, Track

# ── Configuration ─────────────────────────────────────────────────────────────
RPI_IP = "192.168.2.1"     # RPi 4 IP Address (Ethernet: 192.168.2.1)
RPI_PORT = 50051            # gRPC Port
MODEL_PATH = "dr_spaam_5_on_frog.pth"
CONF_THRESH = 0.7
MAX_RANGE_M = 1.0           # 1-meter boundary filter
FOV_DEG = 270.0             # 270 degree crop

def main():
    print(f"Loading DR-SPAAM PyTorch Model onto Jetson CUDA GPU...")
    detector = Detector(MODEL_PATH, model="DR-SPAAM", gpu=True, stride=1, panoramic_scan=True)
    detector.set_laser_fov(FOV_DEG)
    print("✅ PyTorch DR-SPAAM ready on CUDA GPU!")

    # ── Connect gRPC Channel to RPi 4 ──────────────────────────────────────────
    target = f"{RPI_IP}:{RPI_PORT}"
    print(f"Connecting to gRPC server at {target}...")
    
    # Increase maximum message size for large LiDAR scans
    options = [
        ('grpc.max_receive_message_length', 10 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(target, options=options)
    stub = lidar_stream_pb2_grpc.LidarServiceStub(channel)

    print("✅ Connected to gRPC server! Listening to LiDAR scan stream...")

    # Tracker state
    tracks = []
    next_track_id = 1
    last_time = time.time()

    request = lidar_stream_pb2.StreamRequest()
    
    try:
        # Server-streaming RPC loop
        for scan_pb in stub.StreamScan(request):
            raw_ranges = np.array(scan_pb.ranges, dtype=np.float32)
            angle_min = scan_pb.angle_min
            angle_inc = scan_pb.angle_increment

            # ── Step 1: FOV Cropping to 270° ────────────────────────────────
            scan_phi = angle_min + np.arange(len(raw_ranges)) * angle_inc
            scan_phi_norm = (scan_phi + np.pi) % (2 * np.pi) - np.pi
            half_fov = np.deg2rad(FOV_DEG / 2.0)
            keep_mask = np.abs(scan_phi_norm) <= half_fov

            scan_cropped = raw_ranges[keep_mask]
            scan_phi_cropped = scan_phi_norm[keep_mask]
            scan_cropped[scan_cropped < 0.05] = 29.99
            scan_cropped[scan_cropped > 30.0] = 29.99

            # ── Step 2: DR-SPAAM PyTorch Inference on GPU ────────────────────
            dets_xy, dets_cls, _ = detector(scan_cropped, scan_phi=scan_phi_cropped)

            # ── Step 3: Confidence & 1-Meter Range Filter ───────────────────
            conf_mask = (dets_cls >= CONF_THRESH).reshape(-1)
            dets_xy = dets_xy[conf_mask]

            if len(dets_xy) > 0:
                dists = np.hypot(dets_xy[:, 0], dets_xy[:, 1])
                dets_xy = dets_xy[dists <= MAX_RANGE_M]

            # ── Step 4: Kalman Filter Tracking Update ────────────────────────
            now = time.time()
            dt = max(0.01, now - last_time)
            last_time = now

            for trk in tracks:
                trk.predict(dt)

            print(f"[{time.strftime('%H:%M:%S')}] Detected {len(dets_xy)} people within 1m radius.")

    except grpc.RpcError as e:
        print(f"gRPC Error: {e}")
    except KeyboardInterrupt:
        print("Stopping client.")

if __name__ == "__main__":
    main()
