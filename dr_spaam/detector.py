import numpy as np
from dr_spaam.detector import Detector

ckpt = '/home/nilum/proactive-social-nav/forg_dataset/dr_spaam_5_on_frog.pth'
detector = Detector(
    ckpt,
    model="DR-SPAAM",          # Or DROW3
    gpu=True,               # Use GPU
    stride=1,               # Optionally downsample scan for faster inference
    panoramic_scan=True     # Set to True if the scan covers 360 degree
)

# tell the detector field of view of the LiDAR
laser_fov_deg = 360
detector.set_laser_fov(laser_fov_deg)

# detection
num_pts = 1091
import time

print("Starting inference loop. Press Ctrl+C to stop...")
while True:
    # create a random scan (simulating LiDAR ranges)
    scan = np.random.rand(num_pts) * 15.0  # Scales to ~15 meters range

    # detect person
    dets_xy, dets_cls, instance_mask = detector(scan)  # (M, 2), (M,), (N,)
    print(dets_xy, dets_cls, instance_mask)

    # confidence threshold
    cls_thresh = 0.5
    cls_mask = dets_cls > cls_thresh
    dets_xy = dets_xy[cls_mask]
    dets_cls = dets_cls[cls_mask]

    print(f"Scanned {num_pts} points | Detections found: {len(dets_xy)} | Confidence scores: {dets_cls}")
    if len(dets_xy) > 0:
        print(f" -> Coordinates (x, y):\n{dets_xy}")
    
    time.sleep(1.0)
