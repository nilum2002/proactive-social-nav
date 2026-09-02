"""Replay FROG scans through DR-SPAAM + a tracker at a simulated 20 Hz scan
rate, and dump tracks. See config.yaml's header for the decimation method and
its caveats.

Otherwise identical to mot_benchmark_frog/replay.py -- same two pipeline
architectures (sequential / pipelined), same Runner class, same dump schema.
The only change is in main(): the per-segment row index is decimated by
`sampling.decimate` BEFORE anything else touches it, so `scans = scans_all[idx]`
already only contains the kept scans -- `Runner.detect()` and hence the
detector's auto-regressive memory only ever see this decimated stream, exactly
as a genuinely slower sensor would present it. `row`/`ts[row]` in the dumped
output still reference the *original* HDF5 row indices, so score.py just needs
to decimate the same way when selecting which ground-truth rows to score
against -- everything downstream (CLEAR-MOT matching by timestamp) is
unaffected.
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

from common import load_config, DEFAULT_CONFIG

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dr_spaam"))                    # dr_spaam package
sys.path.insert(0, os.path.join(REPO, "ros2_ws", "src", "benchmark"))  # benchmark package

# FROG's native scan period -- used only to seed dt for the first kept scan of
# a segment (prepend value), scaled by the decimation factor below.
NATIVE_DT = 0.025


def build_scan_phi(num_pts, fov_deg):
    """FROG's angle convention -- copied verbatim from mot_benchmark_frog/replay.py."""
    fov = np.radians(fov_deg)
    return np.linspace(-fov / 2, fov / 2, num_pts, endpoint=False).astype(np.float32)


def preprocess(scan, range_min, range_max, pad=29.99):
    """Matches ros2_ws/src/benchmark/benchmark/grpc_server_node.py::_detect."""
    s = np.asarray(scan, dtype=np.float32).copy()
    s[np.isnan(s)] = pad
    s[np.isinf(s)] = pad
    s[s < range_min] = pad
    s[s > range_max] = pad
    return s


class Runner:
    """Shared detection + tracking, independent of pipeline architecture.
    Identical to mot_benchmark_frog/replay.py::Runner -- the rate change lives
    entirely in main()'s scan selection, not here."""

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
        self.detector.set_laser_fov(cfg["frog"]["fov_deg"])
        self.scan_phi = build_scan_phi(cfg["frog"]["num_pts"], cfg["frog"]["fov_deg"])
        self.range_min = cfg["frog"]["range_min"]
        self.range_max = cfg["frog"]["range_max"]
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

    def detect(self, scan):
        if self.reset_every_scan:
            self.reset_detector()
        s = preprocess(scan, self.range_min, self.range_max)
        t0 = time.perf_counter()
        dets_xy, dets_cls, _ = self.detector(s, scan_phi=self.scan_phi)
        self.detect_s.append(time.perf_counter() - t0)
        keep = (dets_cls >= self.conf_thresh).reshape(-1)
        xy = dets_xy[keep]
        return [(float(x), float(y)) for x, y in xy]

    def run_sequential(self, scans, dts):
        self.reset_detector()
        tracker = self._new_tracker()
        out = []
        for i, (scan, dt) in enumerate(zip(scans, dts)):
            dets = self.detect(scan)
            t0 = time.perf_counter()
            active = tracker.step(dt, dets)
            self.track_s.append(time.perf_counter() - t0)
            out.append([(t.id, *t.position) for t in active])
        return out

    def run_pipelined(self, scans, dts):
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
            for i, (scan, dt) in enumerate(zip(scans, dts)):
                work_q.put((i, self.detect(scan), dt))
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
                    help="limit segments (smoke tests)")
    ap.add_argument("--tracker", choices=["kf", "norfair"], help="override tracker.backend")
    ap.add_argument("--weights", help="override detector.weight_file")
    ap.add_argument("--model", choices=["DR-SPAAM", "DROW3"], help="override detector.model")
    ap.add_argument("--reset-every-scan", dest="rse", action="store_true", default=None)
    args = ap.parse_args(argv)

    import h5py
    cfg = load_config(args.config)
    if args.tracker: cfg["tracker"]["backend"] = args.tracker
    if args.weights: cfg["detector"]["weight_file"] = args.weights
    if args.model:   cfg["detector"]["model"] = args.model
    if args.rse is not None: cfg["detector"]["reset_every_scan"] = args.rse
    p = cfg["paths"]
    decimate = int(cfg["sampling"]["decimate"])

    if not os.path.exists(p["gt"]):
        sys.exit(f"missing {p['gt']} -- run mot_benchmark_frog/frog_gt.py first")
    gt = np.load(p["gt"], allow_pickle=True)
    seg, ts = gt["segment"], gt["timestamp"]

    with h5py.File(p["h5"], "r") as f:
        n_h5 = f["timestamps"].shape[0]
        if n_h5 != len(seg):
            sys.exit(f"HDF5 has {n_h5} scans but GT has {len(seg)} -- "
                     f"re-run mot_benchmark_frog/frog_gt.py")
        scans_all = f["scans"][:]

    seg_ids = np.unique(seg)
    if args.max_segments:
        seg_ids = seg_ids[:args.max_segments]
    modes = ["sequential", "pipelined"] if args.pipeline == "both" else [args.pipeline]
    confs = args.conf or cfg["detector"]["conf_thresholds"]

    os.makedirs(p["out_dir"], exist_ok=True)
    for conf in confs:
        runner = Runner(cfg, conf)
        for mode in modes:
            rows, t_start = [], time.perf_counter()
            for s in seg_ids:
                idx = np.nonzero(seg == s)[0]
                idx = idx[::decimate]          # <-- the actual rate change
                if len(idx) == 0:
                    continue
                scans = scans_all[idx]
                dt = np.diff(ts[idx], prepend=ts[idx][0] - NATIVE_DT * decimate)
                dt = np.clip(dt, 1e-3, 2.0)
                fn = runner.run_sequential if mode == "sequential" else runner.run_pipelined
                for row, tracks in zip(idx, fn(scans, dt)):
                    for tid, x, y in (tracks or []):
                        rows.append((s, row, ts[row], tid, x, y))
            arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6))
            out = os.path.join(
                p["out_dir"], f"tracks_{p['bag']}_{runner.tag}_{mode}_conf{conf}.npz")
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
