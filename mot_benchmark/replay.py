"""Replay FROG scans through DR-SPAAM + the Kalman tracker and dump tracks.

Runs entirely outside ROS and gRPC. `MultiObjectTracker` is already decoupled
from both (see its module docstring), and `Detector` never needed them, so the
deployed tracking maths can be exercised here verbatim -- no rclpy, no network,
and deterministic.

Two pipeline architectures are reproduced, mirroring the two ROS nodes:

    sequential  detect -> transform -> tracker.step, inline per scan
                (grpc_server_node.PerceptionServicer)
    pipelined   stage 1 detect on the producer thread, stage 2 tracker.step on
                a consumer thread behind a bounded queue
                (grpc_pipelined_node.PipelinedPerceptionServicer)

Note what this can and cannot show. The pipelined node feeds a *single-consumer*
queue with blocking puts and never drops a frame, so stage 2 sees exactly the
same detections, in the same order, with the same dt as the sequential node.
Offline the two should therefore produce identical tracks and identical MOTA /
MOTP -- pipelining buys throughput and latency, not accuracy. Running both is
how that gets demonstrated rather than assumed; any divergence is a bug worth
finding, not a result.
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import yaml

from frog_gt import load_config, DEFAULT_CONFIG

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dr_spaam"))                    # dr_spaam package
sys.path.insert(0, os.path.join(REPO, "ros2_ws", "src", "benchmark"))  # benchmark package


def build_scan_phi(num_pts, fov_deg):
    """FROG's angle convention, from the official utils/data_loader.py.

    `endpoint=False` matters: it is what the published leaderboard numbers were
    computed with. (The authors' drspaam-frog-support.patch omits it in its own
    frog_handle, a one-bin 0.25 deg discrepancy between their two files.)
    """
    fov = np.radians(fov_deg)
    return np.linspace(-fov / 2, fov / 2, num_pts, endpoint=False).astype(np.float32)


def preprocess(scan, range_min, range_max, pad=29.99):
    """Match ros2_ws/src/benchmark/benchmark/grpc_server_node.py::_detect.

    Kept deliberately identical to the deployed node so the offline numbers
    describe the shipped pipeline. If the AP reproduction step disagrees with
    the FROG leaderboard, this is the first thing to revisit.
    """
    s = np.asarray(scan, dtype=np.float32).copy()
    s[np.isnan(s)] = pad
    s[np.isinf(s)] = pad
    s[s < range_min] = pad
    s[s > range_max] = pad
    return s


class Runner:
    """Shared detection + tracking, independent of pipeline architecture."""

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
        # Names the run in the output filename and the score table. Without it
        # two ablations at the same threshold overwrite each other's dump.
        stem = os.path.splitext(os.path.basename(d["weight_file"]))[0]
        self.variant = stem + ("-nomem" if self.reset_every_scan else "")
        tk = dict(cfg["tracker"])
        self.tracker_backend = str(tk.pop("backend", "kf")).lower()
        self.tracker_kwargs = tk
        # Built here, after the backend is known: the tag namespaces the output
        # dumps so a kf run and a norfair run at the same threshold coexist.
        self.tag = f"{self.variant}_{self.tracker_backend}"
        self.queue_size = cfg["pipeline"]["queue_size"]
        self.detect_s, self.track_s = [], []

    def _new_tracker(self):
        """Build the configured tracker, passing only the keys it accepts.

        Both backends' parameters share one config block, so each constructor is
        filtered against its own signature -- MultiObjectTracker has no **kwargs
        and would raise on norfair's c_init/c_del.
        """
        import inspect
        if self.tracker_backend == "norfair":
            from benchmark.norfair_tracker import NorfairMultiObjectTracker as cls
        else:
            from benchmark.kalman_tracker import MultiObjectTracker as cls
        accepted = set(list(inspect.signature(cls.__init__).parameters)[1:])
        kw = {k: v for k, v in self.tracker_kwargs.items() if k in accepted}
        return cls(**kw)

    def reset_detector(self):
        """Clear DR-SPAAM's temporal memory.

        DR-SPAAM (T > 1) is a *stateful* detector: `Detector.__call__` invokes
        the model with inference=True, and in that mode the spatial-attention
        gate deliberately never resets -- it carries an auto-regressive memory
        (`_memory = alpha * x_new + (1 - alpha) * atten_memory`) from one scan
        to the next. That is the point of the model, but it means the detector
        must be reset wherever the scan sequence is not continuous:

          * at every segment boundary -- strip_empty leaves gaps of up to 81 s
            in the FROG test split, and memory from before a gap is meaningless
            after it;
          * between pipeline variants, so the pipelined run does not begin with
            whatever state the sequential run happened to leave behind.

        Without this, replaying the same scans twice gives slightly different
        detections and the two pipelines appear to disagree at ~1e-7 when in
        fact they are identical.
        """
        gate = getattr(getattr(self.detector, "_model", None), "gate", None)
        if gate is not None and hasattr(gate, "reset"):
            gate.reset()          # DROW3 has no gate; nothing to do there

    def detect(self, scan):
        if self.reset_every_scan:
            # Memoryless: _memory becomes x_new each call, so a T=5 checkpoint
            # runs with no temporal context while the weights stay fixed.
            self.reset_detector()
        s = preprocess(scan, self.range_min, self.range_max)
        t0 = time.perf_counter()
        dets_xy, dets_cls, _ = self.detector(s, scan_phi=self.scan_phi)
        self.detect_s.append(time.perf_counter() - t0)
        keep = (dets_cls >= self.conf_thresh).reshape(-1)
        xy = dets_xy[keep]
        # No odometry is fed, so tracking happens in the sensor frame -- which is
        # the frame FROG's ground-truth circles are natively expressed in. See
        # the note in score.py about ego-motion.
        return [(float(x), float(y)) for x, y in xy]

    # -- architecture 1: sequential ---------------------------------------
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

    # -- architecture 2: pipelined ----------------------------------------
    def run_pipelined(self, scans, dts):
        """Detection on this thread, tracking on a consumer thread.

        Faithful to the ROS node: bounded queue, blocking put (never drop --
        dropping would break dt continuity and corrupt the KF velocities), and a
        single consumer so frame order, and the KF state chain that depends on
        it, are preserved.
        """
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
                except Exception as e:          # must not strand the queue
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
    ap.add_argument("--reset-every-scan", dest="rse", action="store_true", default=None,
                    help="run the checkpoint memoryless (temporal ablation)")
    args = ap.parse_args(argv)

    import h5py
    cfg = load_config(args.config)
    if args.tracker: cfg["tracker"]["backend"] = args.tracker
    if args.weights: cfg["detector"]["weight_file"] = args.weights
    if args.model:   cfg["detector"]["model"] = args.model
    if args.rse is not None: cfg["detector"]["reset_every_scan"] = args.rse
    p = cfg["paths"]
    gt_path = os.path.join(p["out_dir"], f"frog_{p['bag']}_gt.npz")
    if not os.path.exists(gt_path):
        sys.exit(f"missing {gt_path} -- run frog_gt.py first")
    gt = np.load(gt_path, allow_pickle=True)
    seg, ts = gt["segment"], gt["timestamp"]

    with h5py.File(p["h5"], "r") as f:
        n_h5 = f["timestamps"].shape[0]
        if n_h5 != len(seg):
            sys.exit(f"HDF5 has {n_h5} scans but GT has {len(seg)} -- run frog_gt.py "
                     f"without --skip-h5 and resolve the alignment first")
        scans_all = f["scans"][:]

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
                scans = scans_all[idx]
                # dt from the scan clock itself; ~25 ms at FROG's 40 Hz. Within a
                # segment the scans are contiguous, which is exactly why the
                # tracker is rebuilt per segment rather than run across gaps.
                dt = np.diff(ts[idx], prepend=ts[idx][0] - 0.025)
                dt = np.clip(dt, 1e-3, 2.0)
                fn = runner.run_sequential if mode == "sequential" else runner.run_pipelined
                for row, tracks in zip(idx, fn(scans, dt)):
                    for tid, x, y in (tracks or []):
                        rows.append((s, row, ts[row], tid, x, y))
            arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6))
            out = os.path.join(p["out_dir"], f"tracks_{p['bag']}_{runner.tag}_{mode}_conf{conf}.npz")
            np.savez_compressed(
                out, segment=arr[:, 0].astype(np.int64), row=arr[:, 1].astype(np.int64),
                timestamp=arr[:, 2], track_id=arr[:, 3].astype(np.int64),
                xy=arr[:, 4:6], conf_thresh=conf, pipeline=mode, variant=runner.variant,
                tracker=runner.tracker_backend,
                # Which segments this dump actually covers. Scoring must ignore
                # segments that were never replayed (--max-segments), otherwise
                # their ground truth counts as pure misses and MOTA is nonsense.
                segments_run=np.asarray(seg_ids, dtype=np.int64),
            )
            print(f"{runner.tag:>30} {mode:>10} conf={conf:<5} {len(rows):>7} track-obs  "
                  f"{time.perf_counter()-t_start:6.1f}s  -> {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
