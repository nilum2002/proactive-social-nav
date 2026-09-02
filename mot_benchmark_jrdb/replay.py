"""Replay JRDB scans through DR-SPAAM + a tracker and dump tracks.

Structured exactly like mot_benchmark_frog/replay.py -- see that file's
docstring for the sequential-vs-pipelined rationale, which applies unchanged
here. The only real differences: scans are loaded per-row from the paths
jrdb_gt.py recorded (no HDF5), each row is point-reversed + given a
panoramic phi grid (see jrdb_common.py), and one JRDB *sequence* plays the
role FROG's *segment* plays -- the detector's temporal memory is reset once
per sequence, never across a sequence boundary.
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

from jrdb_common import load_config, load_laser, scan_phi_for, preprocess, DEFAULT_CONFIG

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dr_spaam"))
sys.path.insert(0, os.path.join(REPO, "ros2_ws", "src", "benchmark"))


class Runner:
    """Shared detection + tracking, independent of pipeline architecture.
    Mirrors mot_benchmark_frog/replay.py::Runner -- see there for comments
    on reset_detector/detect/run_sequential/run_pipelined; unchanged here."""

    def __init__(self, cfg, conf_thresh):
        from dr_spaam.detector import Detector
        d = cfg["detector"]
        self.conf_thresh = conf_thresh
        self.detector = Detector(
            os.path.join(cfg["paths"]["repo_root"], d["weight_file"])
            if not os.path.isabs(d["weight_file"]) else d["weight_file"],
            model=d["model"], gpu=d["gpu"], stride=d["stride"],
            panoramic_scan=d["panoramic_scan"],
        )
        self.reset_every_scan = bool(d.get("reset_every_scan", False))
        stem = os.path.splitext(os.path.basename(d["weight_file"]))[0]
        self.variant = stem + ("-nomem" if self.reset_every_scan else "")
        tk = dict(cfg["tracker"])
        self.tracker_backend = str(tk.pop("backend", "kf")).lower()
        self.tracker_kwargs = tk
        self.tag = f"{self.variant}_{self.tracker_backend}"
        self.queue_size = cfg["pipeline"]["queue_size"]
        self.detect_s, self.track_s = [], []

    def _new_tracker(self):
        import inspect
        if self.tracker_backend == "norfair":
            from benchmark.norfair_tracker import NorfairMultiObjectTracker as cls
        else:
            from benchmark.kalman_tracker import MultiObjectTracker as cls
        accepted = set(list(inspect.signature(cls.__init__).parameters)[1:])
        kw = {k: v for k, v in self.tracker_kwargs.items() if k in accepted}
        return cls(**kw)

    def reset_detector(self):
        gate = getattr(getattr(self.detector, "_model", None), "gate", None)
        if gate is not None and hasattr(gate, "reset"):
            gate.reset()

    def detect(self, scan, scan_phi):
        if self.reset_every_scan:
            self.reset_detector()
        s = preprocess(scan)
        t0 = time.perf_counter()
        dets_xy, dets_cls, _ = self.detector(s, scan_phi=scan_phi)
        self.detect_s.append(time.perf_counter() - t0)
        keep = (dets_cls >= self.conf_thresh).reshape(-1)
        xy = dets_xy[keep]
        return [(float(x), float(y)) for x, y in xy]

    def run_sequential(self, scans, phis, dts):
        self.reset_detector()
        tracker = self._new_tracker()
        out = []
        for scan, phi, dt in zip(scans, phis, dts):
            dets = self.detect(scan, phi)
            t0 = time.perf_counter()
            active = tracker.step(dt, dets)
            self.track_s.append(time.perf_counter() - t0)
            out.append([(t.id, *t.position) for t in active])
        return out

    def run_pipelined(self, scans, phis, dts):
        self.reset_detector()
        tracker = self._new_tracker()
        work_q = queue.Queue(maxsize=self.queue_size)
        out = [None] * len(scans)
        error = []

        def consumer():
            while True:
                item = work_q.get()
                if item is None:
                    return
                idx, dets, dt = item
                try:
                    t0 = time.perf_counter()
                    active = tracker.step(dt, dets)
                    self.track_s.append(time.perf_counter() - t0)
                    out[idx] = [(t.id, *t.position) for t in active]
                except Exception as e:
                    error.append(e)
                    out[idx] = []

        th = threading.Thread(target=consumer, daemon=True)
        th.start()
        try:
            for i, (scan, phi, dt) in enumerate(zip(scans, phis, dts)):
                work_q.put((i, self.detect(scan, phi), dt))
        finally:
            work_q.put(None)
            th.join(timeout=30.0)
        if error:
            raise error[0]
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--pipeline", choices=["sequential", "pipelined", "both"], default="both")
    ap.add_argument("--conf", type=float, action="append",
                    help="override the swept confidence thresholds")
    ap.add_argument("--max-segments", type=int, default=None,
                    help="limit sequences (smoke tests)")
    ap.add_argument("--tracker", choices=["kf", "norfair"], help="override tracker.backend")
    ap.add_argument("--weights", help="override detector.weight_file")
    ap.add_argument("--model", choices=["DR-SPAAM", "DROW3"], help="override detector.model")
    ap.add_argument("--reset-every-scan", dest="rse", action="store_true", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.tracker: cfg["tracker"]["backend"] = args.tracker
    if args.weights: cfg["detector"]["weight_file"] = args.weights
    if args.model:   cfg["detector"]["model"] = args.model
    if args.rse is not None: cfg["detector"]["reset_every_scan"] = args.rse
    p = cfg["paths"]

    gt_path = os.path.join(p["out_dir"], "jrdb_test_gt.npz")
    if not os.path.exists(gt_path):
        sys.exit(f"missing {gt_path} -- run jrdb_gt.py first")
    gt = np.load(gt_path, allow_pickle=True)
    seg, ts, laser_path = gt["segment"], gt["timestamp"], gt["laser_path"]

    seg_ids = np.unique(seg)
    if args.max_segments:
        seg_ids = seg_ids[:args.max_segments]
    modes = ["sequential", "pipelined"] if args.pipeline == "both" else [args.pipeline]
    confs = args.conf or cfg["detector"]["conf_thresholds"]

    for conf in confs:
        runner = Runner(cfg, conf)
        for mode in modes:
            rows, t_start = [], time.perf_counter()
            for s in seg_ids:
                idx = np.nonzero(seg == s)[0]
                scans = [load_laser(laser_path[i]) for i in idx]
                phis = [scan_phi_for(len(sc)) for sc in scans]
                dt = np.diff(ts[idx], prepend=ts[idx][0] - float(cfg["tracker"]["nominal_dt"]))
                dt = np.clip(dt, 1e-3, 5.0)
                fn = runner.run_sequential if mode == "sequential" else runner.run_pipelined
                for row, tracks in zip(idx, fn(scans, phis, dt)):
                    for tid, x, y in (tracks or []):
                        rows.append((s, row, ts[row], tid, x, y))
            arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6))
            out = os.path.join(p["out_dir"], f"tracks_jrdb_{runner.tag}_{mode}_conf{conf}.npz")
            np.savez_compressed(
                out, segment=arr[:, 0].astype(np.int64), row=arr[:, 1].astype(np.int64),
                timestamp=arr[:, 2], track_id=arr[:, 3].astype(np.int64),
                xy=arr[:, 4:6], conf_thresh=conf, pipeline=mode, variant=runner.variant,
                tracker=runner.tracker_backend,
                segments_run=np.asarray(seg_ids, dtype=np.int64),
            )
            print(f"{runner.tag:>30} {mode:>10} conf={conf:<5} {len(rows):>7} track-obs  "
                  f"{time.perf_counter()-t_start:6.1f}s  -> {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
