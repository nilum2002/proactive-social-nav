"""Shared config loading and DROWv2 raw-file parsing for this directory's scripts.

Deliberately reimplements the loading half of `dr_spaam.datahandle.drow_handle
.DROWHandle` (rather than importing it) because that class is a torch
`Dataset` built around random-access `__getitem__` sampling for training; here
every script wants the *whole* sequence walked in order, so a flatter
"load one sequence, get its arrays" helper is a better fit. The on-disk format
this parses (.csv scans, .wp person annotations, frame-id matching) is
identical, and get_laser_phi() below is copied verbatim from there since it is
a small, load-bearing geometry constant.
"""
import glob
import json
import os

import numpy as np
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path=DEFAULT_CONFIG):
    """Load config.yaml and make every path absolute, resolved against the
    config file's own directory (not the process cwd) so these scripts behave
    the same whether run from the repo root or from inside this directory."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(path)
    root = os.path.abspath(os.path.join(cfg_dir, cfg["paths"].get("repo_root", "..")))
    cfg["paths"]["repo_root"] = root
    for k, v in cfg["paths"].items():
        if k not in ("repo_root", "split"):
            cfg["paths"][k] = v if os.path.isabs(v) else os.path.join(root, v)
    return cfg


def get_laser_phi(angle_inc=np.radians(0.5), num_pts=450):
    """DROW's SICK S300: 225 deg fov, 450 pts, mounted at 37cm height.

    Verbatim from DROWHandle.get_laser_phi -- note this uses linspace's
    default endpoint=True, unlike FROG's build_scan_phi (endpoint=False in
    mot_benchmark/replay.py). Getting this wrong silently misaligns every
    detection's angle by a fraction of a degree.
    """
    laser_fov = (num_pts - 1) * angle_inc
    return np.linspace(-laser_fov * 0.5, laser_fov * 0.5, num_pts).astype(np.float32)


def list_sequences(data_dir, split):
    """Sequence base names (path with extension stripped) for a split, sorted
    for a deterministic iteration order across runs."""
    return sorted(f[:-4] for f in glob.glob(os.path.join(data_dir, split, "*.csv")))


class Sequence:
    """One DROW run: full scan stream plus sparse (r,phi) person annotations.

    `ann_scan_idx[k]` is the index into `scans`/`t` for the k-th annotated
    frame, i.e. the frame `wp[k]` belongs to -- annotations exist for roughly
    1 in 20 scans, not every scan.
    """

    def __init__(self, seq_path):
        self.name = os.path.basename(seq_path)

        data = np.genfromtxt(seq_path + ".csv", delimiter=",")
        self.scan_ns = data[:, 0].astype(np.int64)
        self.t = data[:, 1].astype(np.float64)
        self.scans = data[:, 2:].astype(np.float32)

        ann_ns, wp = self._load_wp(seq_path + ".wp")
        self.ann_ns = ann_ns
        self.wp = wp  # list[np.ndarray[(r, phi), ...]], one per annotated frame

        # map each annotated frame id to its row in scans/t, matching
        # DROWHandle._load_scan_sequence's assumption that both files are
        # sorted ascending by frame id and every ann id appears in the scans.
        is_ = 0
        idx = []
        for ns in ann_ns:
            while self.scan_ns[is_] != ns:
                is_ += 1
            idx.append(is_)
        self.ann_scan_idx = np.array(idx, dtype=np.int64)

    @staticmethod
    def _load_wp(f_name):
        ns, dets = [], []
        with open(f_name) as f:
            for line in f:
                head, tail = line.split(",", 1)
                ns.append(int(head))
                pts = json.loads(tail)
                dets.append(np.asarray(pts, dtype=np.float32).reshape(-1, 2))
        return np.array(ns, dtype=np.int64), dets

    def window(self, scan_idx, num_scans, stride=1):
        """Last `num_scans` scans ending at (and including) `scan_idx`, oldest
        first. Short of history at the sequence start repeats the first scan,
        matching DROWHandle.__getitem__'s `max(0, scan_idx - i)` clamp."""
        delta = (np.arange(num_scans) * stride)[::-1]
        inds = [max(0, scan_idx - int(d)) for d in delta]
        return self.scans[inds]
