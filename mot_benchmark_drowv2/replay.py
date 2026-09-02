"""Replay DROWv2 test-split scans through DR-SPAAM + a tracker, for the
SECONDARY, approximate MOT benchmark (pseudo_gt.py -> replay.py -> score.py).
See pseudo_gt.py's docstring for why this is an approximation, not official
MOT ground truth.

Streaming, persistent-memory inference (Detector.__call__, inference=True) --
i.e. the *deployment* protocol, matching mot_benchmark/replay.py for FROG --
not the windowed/gate-reset protocol drow_ap.py uses to match the published
DROW/DR-SPAAM AP benchmark. The two scripts intentionally use different
protocols because they answer different questions: drow_ap.py reproduces a
literature metric, this measures the tracker atop the same detector runtime
FROG's benchmark and the live ROS node use.

The tracker runs over every scan in a sequence (for realistic continuous
dt/motion), but only the track state at ANNOTATED frame timestamps is
recorded, matching pseudo_gt.py's sparse ground truth -- otherwise every
un-annotated scan's track position would score as a false positive against
ground truth that was simply never collected there, not against a real
detector error.
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

from drow_common import load_config, DEFAULT_CONFIG, list_sequences, Sequence, get_laser_phi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dr_spaam"))                     # dr_spaam package
sys.path.insert(0, os.path.join(REPO, "ros2_ws", "src", "benchmark"))  # benchmark package


def preprocess(scan, pad=29.99):
    s = np.asarray(scan, dtype=np.float32).copy()
    s[np.isnan(s)] = pad
    s[np.isinf(s)] = pad
    return s


class Runner:
    """Shared detection + tracking, independent of pipeline architecture.
    Mirrors mot_benchmark/replay.py::Runner; see that file for rationale on
    the pieces duplicated here (memory reset at segment boundaries, tracker
    filtering by constructor signature, etc.) -- this directory has its own
    copy rather than importing across mot_benchmark/ so the two benchmarks
    can diverge without coupling."""

    def __init__(self, cfg, conf_thresh):
        from dr_spaam.detector import Detector
        d = cfg["detector"]
        p = cfg["paths"]
        self.conf_thresh = conf_thresh
        weight_file = d["weight_file"]
        if not os.path.isabs(weight_file):
            weight_file = os.path.join(p["repo_root"], weight_file)
        self.detector = Detector(
            weight_file, model=d["model"], gpu=d["gpu"], stride=1,
            panoramic_scan=False,
        )
        self.scan_phi = get_laser_phi()
        self.reset_every_scan = bool(d.get("reset_every_scan", False))
        stem = os.path.splitext(os.path.basename(d["weight_file"]))[0]
        self.variant = stem + ("-nomem" if self.reset_every_scan else "")
        tk = dict(cfg["tracker"])
        self.tracker_backend = str(tk.pop("backend", "kf")).lower()
        self.tracker_kwargs = tk
        self.tag = f"{self.variant}_{self.tracker_backend}"
        self.queue_size = cfg["pipeline"]["queue_size"]

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
        s = preprocess(scan)
        dets_xy, dets_cls, _ = self.detector(s, scan_phi=self.scan_phi)
        keep = (dets_cls >= self.conf_thresh).reshape(-1)
        xy = dets_xy[keep]
        return [(float(x), float(y)) for x, y in xy]

    def run_sequential(self, scans, dts, ann_mask):
        self.reset_detector()
        tracker = self._new_tracker()
        out = []
        for scan, dt, is_ann in zip(scans, dts, ann_mask):
            dets = self.detect(scan)
            active = tracker.step(dt, dets)
            if is_ann:
                out.append([(t.id, *t.position) for t in active])
        return out

    def run_pipelined(self, scans, dts, ann_mask):
        self.reset_detector()
        tracker = self._new_tracker()
        work_q = queue.Queue(maxsize=self.queue_size)
        out = []
        error = []

        def consumer():
            while True:
                item = work_q.get()
                if item is None:
                    return
                dets, dt, is_ann = item
                try:
                    active = tracker.step(dt, dets)
                    if is_ann:
                        out.append([(t.id, *t.position) for t in active])
                except Exception as e:
                    error.append(e)

        th = threading.Thread(target=consumer, daemon=True)
        th.start()
        try:
            for scan, dt, is_ann in zip(scans, dts, ann_mask):
                work_q.put((self.detect(scan), dt, is_ann))
        finally:
            work_q.put(None)
            th.join(timeout=60.0)
        if error:
            raise error[0]
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--pipeline", choices=["sequential", "pipelined", "both"], default="both")
    ap.add_argument("--conf", type=float, action="append",
                    help="override the swept confidence thresholds")
    ap.add_argument("--max-sequences", type=int, default=None, help="limit sequences (smoke tests)")
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

    seqs = list_sequences(p["data_dir"], p["split"])
    if args.max_sequences:
        seqs = seqs[: args.max_sequences]
    if not seqs:
        sys.exit(f"no sequences found under {p['data_dir']}/{p['split']}")

    modes = ["sequential", "pipelined"] if args.pipeline == "both" else [args.pipeline]
    confs = args.conf or cfg["detector"]["conf_thresholds"]

    for conf in confs:
        runner = Runner(cfg, conf)
        for mode in modes:
            rows, t_start = [], time.perf_counter()
            for seg, seq_path in enumerate(seqs):
                seq = Sequence(seq_path)
                ann_set = set(int(i) for i in seq.ann_scan_idx)
                ann_mask = [i in ann_set for i in range(len(seq.scans))]
                if len(seq.t) > 1:
                    median_dt = float(np.median(np.diff(seq.t)))
                else:
                    median_dt = 0.1
                dt = np.diff(seq.t, prepend=seq.t[0] - median_dt)
                dt = np.clip(dt, 1e-3, 2.0)

                fn = runner.run_sequential if mode == "sequential" else runner.run_pipelined
                tracks_per_ann_frame = fn(seq.scans, dt, ann_mask)
                # ann_scan_idx is sorted ascending (same order the .wp file was
                # read in), so tracks_per_ann_frame[k] lines up with
                # seq.ann_ns[k] / seq.t[seq.ann_scan_idx[k]] positionally.
                for k, tracks in enumerate(tracks_per_ann_frame):
                    ts = float(seq.t[seq.ann_scan_idx[k]])
                    for tid, x, y in (tracks or []):
                        rows.append((seg, k, ts, tid, x, y))

            arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6))
            out_dir = p["out_dir"]
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(
                out_dir, f"drowv2_{p['split']}_tracks_{runner.tag}_{mode}_conf{conf}.npz")
            np.savez_compressed(
                out, segment=arr[:, 0].astype(np.int64), row=arr[:, 1].astype(np.int64),
                timestamp=arr[:, 2], track_id=arr[:, 3].astype(np.int64),
                xy=arr[:, 4:6], conf_thresh=conf, pipeline=mode, variant=runner.variant,
                tracker=runner.tracker_backend,
                segments_run=np.arange(len(seqs), dtype=np.int64),
            )
            print(f"{runner.tag:>30} {mode:>10} conf={conf:<5} {len(rows):>6} track-obs  "
                  f"{time.perf_counter()-t_start:6.1f}s  -> {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
