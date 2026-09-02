"""Shared config loading and FROG ground-truth/scan loading for this
directory's scripts. Ground truth itself is built by mot_benchmark_frog/frog_gt.py
(not reimplemented here) -- this just reads that npz plus the matching HDF5
scans, grouped by segment.
"""
import os

import h5py
import numpy as np
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path=DEFAULT_CONFIG):
    """Load config.yaml and make every path absolute, resolved against the
    config file's own directory (not the process cwd)."""
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


def build_scan_phi(num_pts, fov_deg):
    """FROG's angle convention, from the official utils/data_loader.py.

    `endpoint=False` matters: it is what the published leaderboard numbers
    were computed with. Copied verbatim from mot_benchmark_frog/replay.py so both
    benchmarks stay bit-identical on this.
    """
    fov = np.radians(fov_deg)
    return np.linspace(-fov / 2, fov / 2, num_pts, endpoint=False).astype(np.float32)


def preprocess(scan, range_min, range_max, pad=29.99):
    """Matches ros2_ws/src/benchmark/benchmark/grpc_server_node.py::_detect
    and mot_benchmark_frog/replay.py::preprocess -- kept identical so this
    describes the same pipeline that's actually deployed."""
    s = np.asarray(scan, dtype=np.float32).copy()
    s[np.isnan(s)] = pad
    s[np.isinf(s)] = pad
    s[s < range_min] = pad
    s[s > range_max] = pad
    return s


class GroundTruth:
    """Loads frog_gt.py's npz plus the matching HDF5 scans, grouped by
    segment (a temporally contiguous run; frog_gt.py breaks a new one after
    any gap -- crossing that gap to build a T-scan window would mix in scans
    from an unrelated moment)."""

    def __init__(self, gt_path, h5_path):
        gt = np.load(gt_path, allow_pickle=True)
        with h5py.File(h5_path, "r") as f:
            n_h5 = f["timestamps"].shape[0]
            if n_h5 != len(gt["segment"]):
                raise ValueError(
                    f"HDF5 has {n_h5} scans but GT has {len(gt['segment'])} -- "
                    f"re-run mot_benchmark_frog/frog_gt.py")
            self.scans_all = f["scans"][:]

        self.timestamp = gt["timestamp"]
        self.segment = gt["segment"]
        self.gt_scan = gt["gt_scan"]
        self.gt_xy = gt["gt_xy"]
        self.seg_ids = np.unique(self.segment)

    def segment_rows(self, seg):
        """Row indices (into scans_all/timestamp) for one segment, in order."""
        return np.nonzero(self.segment == seg)[0]

    def points_for_row(self, row):
        m = self.gt_scan == row
        return self.gt_xy[m]

    @staticmethod
    def window(scans, local_idx, num_scans):
        """Last `num_scans` scans ending at (and including) `local_idx` within
        one segment's own scan array, oldest first. Short history at a
        segment's start repeats the first scan."""
        delta = np.arange(num_scans)[::-1]
        inds = [max(0, local_idx - int(d)) for d in delta]
        return scans[inds]
