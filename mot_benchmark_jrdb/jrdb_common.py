"""Minimal JRDB sequence loader for the MOT benchmark.

Deliberately NOT dr_spaam.datahandle.JRDBHandle: that class unconditionally
loads stitched images and 3D pointclouds per frame (`_LOAD_PC_IM = True`,
hardcoded at module scope), which this benchmark never needs -- it is
2D-lidar detection + tracking only. Reading just timestamps/, labels_3d/ and
lasers/ means the images/pointclouds/detections_2d_stitched JRDB subsets
don't need to be downloaded at all for this benchmark.
"""
import json
import os

import numpy as np
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(REPO, "dr_spaam"))                    # dr_spaam package
sys.path.insert(0, os.path.join(REPO, "ros2_ws", "src", "benchmark"))  # benchmark package

import dr_spaam.utils.jrdb_transforms as jt  # noqa: E402

# The DR-SPAAM authors' own held-out split (dr_spaam/dr_spaam/dataset/jrdb_dataset.py,
# _JRDB_TEST_SEQUENCES) -- copied literally rather than imported, since that
# name is underscore-prefixed (not a public API) and we want this list to be
# stable even if jrdb_dataset.py's internals change. These sequences live
# under train_dataset/ (JRDB's own test_dataset/ has no public 3D labels --
# it is held out for the official leaderboard, not local evaluation).
TEST_SEQUENCES = [
    "packard-poster-session-2019-03-20_1",
    "gates-to-clark-2019-02-28_1",
    "packard-poster-session-2019-03-20_0",
    "tressider-2019-03-16_1",
    "clark-center-2019-02-28_0",
    "svl-meeting-gates-2-2019-04-08_1",
    "meyer-green-2019-03-16_0",
    "gates-159-group-meeting-2019-04-03_0",
    "huang-2-2019-01-25_0",
    "gates-ai-lab-2019-02-08_0",
]


def load_config(path=DEFAULT_CONFIG):
    """Load the config and make every path absolute, relative to the config
    file's own directory -- same convention as the other mot_benchmark_*/
    load_config functions, so these scripts behave the same run from
    anywhere."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(path)
    root = os.path.abspath(os.path.join(cfg_dir, cfg["paths"].get("repo_root", "..")))
    cfg["paths"]["repo_root"] = root
    for k, v in cfg["paths"].items():
        if k != "repo_root":
            cfg["paths"][k] = v if os.path.isabs(v) else os.path.join(root, v)
    return cfg


def _to_drow_xy(cx, cy, cz):
    """3D box center (base frame) -> 2D (x, y) in the DROW/DR-SPAAM training
    frame. Mirrors dr_spaam/dr_spaam/dataset/jrdb_dataset.py::_get_sample's
    annotation path exactly (base -> laser, then flip y)."""
    xyz = np.array([[cx], [cy], [cz]], dtype=np.float32)
    xyz = jt.transform_pts_base_to_laser(xyz)
    return float(xyz[0, 0]), float(-xyz[1, 0])


def list_sequence_frames(data_dir, seq):
    """Labeled frames of one JRDB sequence, lidar + real GT identity only.

    Returns a list of dicts, one per labeled frame, in file order:
        frame_id, timestamp, laser_path (absolute), anns: [(label_id, x, y)]
    `label_id` is JRDB's own persistent string id (e.g. "pedestrian:46"),
    consistent within this sequence -- real ground truth, no linking needed.
    """
    train_dir = os.path.join(data_dir, "train_dataset")

    ts_path = os.path.join(train_dir, "timestamps", seq, "frames_pc_im_laser.json")
    with open(ts_path) as f:
        frames = json.load(f)["data"]

    label_path = os.path.join(train_dir, "labels", "labels_3d", f"{seq}.json")
    with open(label_path) as f:
        pc_labels = json.load(f)["labels"]

    out = []
    for frame in frames:
        pc_file = os.path.basename(frame["pc_frame"]["pointclouds"][0]["url"])
        if pc_file not in pc_labels:
            continue  # unlabeled frame -- not usable as ground truth
        anns = [
            (str(ann["label_id"]), *_to_drow_xy(
                float(ann["box"]["cx"]), float(ann["box"]["cy"]), float(ann["box"]["cz"])))
            for ann in pc_labels[pc_file]
        ]
        out.append(dict(
            frame_id=int(frame["frame_id"]),
            timestamp=float(frame["laser_frame"]["timestamp"]),
            laser_path=os.path.join(train_dir, frame["laser_frame"]["url"]),
            anns=anns,
        ))
    out.sort(key=lambda d: d["timestamp"])
    return out


def load_laser(path):
    """One combined panoramic scan. Point-reversed to match training: JRDB's
    laser frame is x-forward/y-left, DR-SPAAM/DROW was trained x-forward/
    y-right -- reversing point order is the discrete equivalent of negating
    phi for a symmetric linspace(-pi, pi) grid (see jrdb_dataset.py's
    `laser_data[:, ::-1]`). Getting this backwards silently mirrors every
    detection left-right; it will not raise an error."""
    scan = np.loadtxt(path, dtype=np.float32)
    return scan[::-1]


def scan_phi_for(n_pts):
    """JRDB's own convention: endpoint=True (unlike FROG's endpoint=False),
    so -pi and +pi are both present (the wrap-around point is duplicated).
    Kept faithful to what the checkpoint was trained with, not "fixed"."""
    return np.linspace(-np.pi, np.pi, n_pts, dtype=np.float32)


def preprocess(scan, pad=29.99):
    s = np.asarray(scan, dtype=np.float32).copy()
    s[np.isnan(s)] = pad
    s[np.isinf(s)] = pad
    return s
