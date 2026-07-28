#!/usr/bin/env python3
"""gRPC LiDAR OpenCV Visualizer (Jetson Nano Standalone)

Connects to Raspberry Pi 4 gRPC LiDAR stream, performs DR-SPAAM GPU inference,
runs Kalman Filter tracking (CV model + Hungarian matching + static filter),
and renders a real-time OpenCV 2D Bird's-Eye View (BEV) visualization window.

Replicates RViz markers:
 - Cyan Circle: Tracked Person (ID)
 - Yellow Arrow: Velocity Vector (speed & direction)
 - Green Path & Sphere: Predicted Future Position (1.5s horizon)
 - Red Ring: 1-Meter Range Boundary
"""

import sys
import os
import time
import numpy as np
import cv2
import grpc

# Import generated Protobuf files and tracker components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lidar_stream_pb2
import lidar_stream_pb2_grpc

from dr_spaam.detector import Detector
from dr_spaam_tracker_node import KalmanFilter, Track, HAS_SCIPY
if HAS_SCIPY:
    from scipy.optimize import linear_sum_assignment

# ── Network & Detector Config ──────────────────────────────────────────────────
RPI_IP = "192.168.2.1"          # RPi 4 IP Address (Ethernet)
RPI_PORT = 50051                 # gRPC Port
MODEL_PATH = "/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/frog_dataset/dr_spaam_5_on_frog.pth"
CONF_THRESH = 0.7
MAX_RANGE_M = 2                  # 1-meter detection boundary
FOV_DEG = 270.0                  # 270° scan crop

# ── Tracker Config ─────────────────────────────────────────────────────────────
ASSOC_THRESH = 0.8               # Max association matching distance (m)
MAX_LOST_FRAMES = 10             # Max lost frames before track deletion
PRED_HORIZON = 1.5               # Prediction horizon (seconds)
STATIC_SPEED_THRESH = 0.15       # Speed below this (m/s) is static
STATIC_FRAMES_REQ = 15           # Static frames required before removal
STD_A_X = 1.5
STD_A_Y = 1.5
R_LASER = 0.20

# ── OpenCV Window Config ───────────────────────────────────────────────────────
WIN_SIZE = 800                   # Canvas size (pixels)
MAX_DISPLAY_RANGE = 2.5          # Maximum radius displayed on canvas (meters)
SCALE = WIN_SIZE / (2 * MAX_DISPLAY_RANGE)  # Pixels per meter
CENTER = (WIN_SIZE // 2, WIN_SIZE // 2)     # Robot/LiDAR is at center (400, 400)


def world_to_img(x, y):
    """Convert 2D world coordinates (x-forward, y-left) to OpenCV image coordinates (u-right, v-down)."""
    # World: +X is forward (up on screen), +Y is left (left on screen)
    u = int(CENTER[0] - y * SCALE)
    v = int(CENTER[1] - x * SCALE)
    return (u, v)


def render_bev(raw_ranges_cropped, scan_phi_cropped, tracks, fps):
    """Render 2D Bird's-Eye View image using OpenCV."""
    canvas = np.zeros((WIN_SIZE, WIN_SIZE, 3), dtype=np.uint8)

    # 1. Draw Grid lines (0.5m intervals)
    for r_m in np.arange(0.5, MAX_DISPLAY_RANGE + 0.1, 0.5):
        r_px = int(r_m * SCALE)
        color = (40, 40, 40) if r_m != MAX_RANGE_M else (0, 0, 180)  # Red for 1m boundary
        thickness = 1 if r_m != MAX_RANGE_M else 2
        cv2.circle(canvas, CENTER, r_px, color, thickness)
        cv2.putText(canvas, f"{r_m:.1f}m", (CENTER[0] + 5, CENTER[1] - r_px + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # Crosshairs (LiDAR origin)
    cv2.line(canvas, (CENTER[0] - 15, CENTER[1]), (CENTER[0] + 15, CENTER[1]), (80, 80, 80), 1)
    cv2.line(canvas, (CENTER[0], CENTER[1] - 15), (CENTER[0], CENTER[1] + 15), (80, 80, 80), 1)
    cv2.circle(canvas, CENTER, 6, (0, 255, 255), -1)  # Yellow center dot (Robot/LiDAR)

    # 2. Draw raw LiDAR point cloud
    valid_mask = raw_ranges_cropped < 29.0
    valid_ranges = raw_ranges_cropped[valid_mask]
    valid_phis = scan_phi_cropped[valid_mask]

    pts_x = valid_ranges * np.cos(valid_phis)
    pts_y = valid_ranges * np.sin(valid_phis)

    for px, py in zip(pts_x, pts_y):
        u, v = world_to_img(px, py)
        if 0 <= u < WIN_SIZE and 0 <= v < WIN_SIZE:
            canvas[v, u] = (180, 180, 180)

    # 3. Draw 1-Meter Range Boundary Label
    u_b, v_b = world_to_img(MAX_RANGE_M, 0)
    cv2.putText(canvas, "1m BOUNDARY", (u_b - 45, v_b - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 4. Draw Active Tracked Pedestrians
    for track in tracks:
        if track.state != "ACTIVE" or track.is_static:
            continue

        tx, ty = track.position
        vx, vy = track.velocity
        speed = track.speed
        u_trk, v_trk = world_to_img(tx, ty)

        # 4a. Future Prediction (Green Line & Ghost Circle)
        future_x = tx + vx * PRED_HORIZON
        future_y = ty + vy * PRED_HORIZON
        u_fut, v_fut = world_to_img(future_x, future_y)

        # Green dashed line to predicted future position
        cv2.line(canvas, (u_trk, v_trk), (u_fut, v_fut), (0, 255, 0), 2)
        # Green predicted future sphere
        cv2.circle(canvas, (u_fut, v_fut), int(0.2 * SCALE), (0, 200, 0), 1)

        # 4b. Velocity Arrow (Yellow)
        if speed > 0.1:
            u_vel, v_vel = world_to_img(tx + vx * 0.8, ty + vy * 0.8)
            cv2.arrowedLine(canvas, (u_trk, v_trk), (u_vel, v_vel), (0, 230, 255), 2, tipLength=0.3)

        # 4c. Current Person Circle (Cyan)
        cv2.circle(canvas, (u_trk, v_trk), int(0.25 * SCALE), (255, 255, 0), 2)
        cv2.circle(canvas, (u_trk, v_trk), 3, (255, 255, 0), -1)

        # 4d. Label Text (ID, Speed, Distance)
        dist = np.hypot(tx, ty)
        label = f"ID:{track.id} | {speed:.2f}m/s | {dist:.2f}m"
        cv2.putText(canvas, label, (u_trk - 40, v_trk - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # 5. Overlay Stats Panel
    cv2.rectangle(canvas, (10, 10), (320, 75), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, 10), (320, 75), (100, 100, 100), 1)
    cv2.putText(canvas, "DR-SPAAM Pedestrian Tracking (BEV)", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(canvas, f"FPS: {fps:.1f}  |  Active Tracks: {len([t for t in tracks if t.state=='ACTIVE'])}",
                (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, f"Boundary: {MAX_RANGE_M:.1f}m  |  FOV: {FOV_DEG}°",
                (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return canvas


def main():
    print(f"Loading DR-SPAAM PyTorch Model onto GPU...")
    detector = Detector(MODEL_PATH, model="DR-SPAAM", gpu=True, stride=1, panoramic_scan=True)
    detector.set_laser_fov(FOV_DEG)
    print("✅ PyTorch DR-SPAAM ready on CUDA GPU!")

    # Connect to gRPC stream
    target = f"{RPI_IP}:{RPI_PORT}"
    print(f"Connecting to gRPC server at {target}...")
    options = [
        ('grpc.max_receive_message_length', 10 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
    ]
    channel = grpc.insecure_channel(target, options=options)
    stub = lidar_stream_pb2_grpc.LidarServiceStub(channel)

    print("✅ Connected to gRPC stream! Opening OpenCV Visualizer...")

    # Tracker state
    tracks = []
    next_track_id = 1
    last_time = time.time()
    fps_counter = 0
    fps_start = time.time()
    fps = 0.0

    static_blacklist = []

    request = lidar_stream_pb2.StreamRequest()

    cv2.namedWindow("DR-SPAAM Pedestrian Tracking (BEV)", cv2.WINDOW_AUTOSIZE)

    try:
        for scan_pb in stub.StreamScan(request):
            raw_ranges = np.array(scan_pb.ranges, dtype=np.float32)
            angle_min = scan_pb.angle_min
            angle_inc = scan_pb.angle_increment

            # ── 1. FOV Cropping ─────────────────────────────────────────────
            scan_phi = angle_min + np.arange(len(raw_ranges)) * angle_inc
            scan_phi_norm = (scan_phi + np.pi) % (2 * np.pi) - np.pi
            half_fov = np.deg2rad(FOV_DEG / 2.0)
            keep_mask = np.abs(scan_phi_norm) <= half_fov

            scan_cropped = raw_ranges[keep_mask]
            scan_phi_cropped = scan_phi_norm[keep_mask]
            scan_cropped[scan_cropped < 0.05] = 29.99
            scan_cropped[scan_cropped > 30.0] = 29.99

            # ── 2. DR-SPAAM PyTorch GPU Inference ────────────────────────────
            dets_xy, dets_cls, _ = detector(scan_cropped, scan_phi=scan_phi_cropped)

            # ── 3. Confidence & 1-Meter Range Filter ────────────────────────
            conf_mask = (dets_cls >= CONF_THRESH).reshape(-1)
            dets_xy = dets_xy[conf_mask]

            if len(dets_xy) > 0:
                dists = np.hypot(dets_xy[:, 0], dets_xy[:, 1])
                dets_xy = dets_xy[dists <= MAX_RANGE_M]

            # ── 4. Kalman Filter Multi-Object Tracking ────────────────────────
            now = time.time()
            dt = max(0.01, now - last_time)
            last_time = now

            for trk in tracks:
                trk.predict(dt)

            detections = [(d[0], d[1]) for d in dets_xy]
            matched_dets = set()
            matched_trks = set()
            associations = []

            if len(tracks) > 0 and len(detections) > 0:
                if HAS_SCIPY:
                    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
                    for t_i, trk in enumerate(tracks):
                        for d_i, det in enumerate(detections):
                            cost_matrix[t_i, d_i] = np.hypot(trk.position[0] - det[0], trk.position[1] - det[1])
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] < ASSOC_THRESH:
                            associations.append((r, c))
                            matched_trks.add(r)
                            matched_dets.add(c)

            for t_i, d_i in associations:
                tracks[t_i].update(detections[d_i])

            for t_i, trk in enumerate(tracks):
                if t_i not in matched_trks:
                    trk.lost_count += 1

            # Expire blacklist
            static_blacklist = [(bx, by, exp) for (bx, by, exp) in static_blacklist if now < exp]

            for d_i, det in enumerate(detections):
                if d_i not in matched_dets:
                    is_blacklisted = any(np.hypot(det[0] - bx, det[1] - by) < ASSOC_THRESH for (bx, by, _) in static_blacklist)
                    if not is_blacklisted:
                        new_track = Track(next_track_id, det, STD_A_X, STD_A_Y, R_LASER)
                        tracks.append(new_track)
                        next_track_id += 1

            active_tracks = []
            for trk in tracks:
                trk.check_static(STATIC_SPEED_THRESH, STATIC_FRAMES_REQ)
                if trk.lost_count <= MAX_LOST_FRAMES and not trk.is_static:
                    active_tracks.append(trk)
                elif trk.is_static:
                    static_blacklist.append((trk.position[0], trk.position[1], now + 5.0))
            tracks = active_tracks

            # ── 5. FPS Counter ───────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            # ── 6. Render & Display OpenCV BEV ──────────────────────────────
            frame = render_bev(scan_cropped, scan_phi_cropped, tracks, fps)
            cv2.imshow("DR-SPAAM Pedestrian Tracking (BEV)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    except grpc.RpcError as e:
        print(f"gRPC Error: {e}")
    except KeyboardInterrupt:
        print("Stopping visualizer.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
