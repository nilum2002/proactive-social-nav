"""Score replayed 20 Hz-simulated FROG tracks against ground truth with CLEAR
MOT (MOTA / MOTP). See config.yaml's header for the decimation method and its
caveats. Otherwise identical to mot_benchmark_frog/score.py -- same
ClearMotAssociator-based accounting, same Stone Soup 1.9.1 zero-length
TimeRange patch, same deterministic-ordering fix for build_truths/build_tracks
(both apply here for the same reasons, unrelated to scan rate).

The only real difference: ground truth rows are decimated the same way
replay.py decimated which scans got detected+tracked (`sampling.decimate` in
config.yaml), via `build_truths`'s existing `rows` filter argument -- no
changes needed to the CLEAR-MOT accounting itself. Without this, scoring would
count every scan replay.py deliberately skipped as a pure miss, which would
measure "detector randomly drops half its frames" rather than "detector only
ever sees a 20 Hz stream".
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from common import load_config, DEFAULT_CONFIG

from stonesoup.types.track import Track
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import State
from stonesoup.types.array import StateVector
from stonesoup.measures import Euclidean
from stonesoup.dataassociator.clearmot import ClearMotAssociator
from stonesoup.metricgenerator.clearmotmetrics import ClearMotMetrics
from stonesoup.metricgenerator.manager import MultiManager
from stonesoup.types.interval import Interval

UTC = dt.timezone.utc

# Stone Soup 1.9.1 bug -- see mot_benchmark_frog/score.py's identical comment.
_orig_interval_init = Interval.__init__


def _patched_interval_init(self, *args, **kwargs):
    try:
        _orig_interval_init(self, *args, **kwargs)
    except ValueError:
        bound = dict(zip(("start", "end"), args)) | kwargs
        start, end = bound["start"], bound["end"]
        if isinstance(start, dt.datetime) and start == end:
            bound["end"] = end + dt.timedelta(microseconds=1)
            _orig_interval_init(self, **bound)
        else:
            raise


Interval.__init__ = _patched_interval_init


def _ts(x):
    return dt.datetime.fromtimestamp(round(float(x), 6), tz=UTC)


def build_truths(gt, rows):
    """One GroundTruthPath per FROG tracklet id present in these rows.

    `rows` is the decimated (20 Hz) row set passed in from main() -- see
    module docstring. Returns a list sorted by id, not a set -- see
    mot_benchmark_frog/score.py's build_truths docstring for why.
    """
    by_id = defaultdict(list)
    row_set = set(rows.tolist())
    for scan_row, gid, (x, y) in zip(gt["gt_scan"], gt["gt_id"], gt["gt_xy"]):
        if int(scan_row) not in row_set:
            continue
        by_id[int(gid)].append((_ts(gt["timestamp"][int(scan_row)]), float(x), float(y)))
    truths = []
    for gid, obs in sorted(by_id.items()):
        obs.sort(key=lambda o: o[0])
        truths.append(GroundTruthPath(
            [GroundTruthState(StateVector([x, y]), timestamp=t) for t, x, y in obs],
            id=f"gt{gid}"))
    return truths


def build_tracks(tr, seg):
    """One Track per tracker id within this segment. Already only contains the
    decimated rows replay.py recorded -- see build_truths."""
    m = tr["segment"] == seg
    by_id = defaultdict(list)
    for tid, t, (x, y) in zip(tr["track_id"][m], tr["timestamp"][m], tr["xy"][m]):
        by_id[int(tid)].append((_ts(t), float(x), float(y)))
    tracks = []
    for tid, obs in sorted(by_id.items()):
        obs.sort(key=lambda o: o[0])
        tracks.append(Track([State(StateVector([x, y]), timestamp=t) for t, x, y in obs],
                            id=f"s{seg}t{tid}"))
    return tracks


def clearmot_counts(tracks, truths, assoc_set):
    """Bernardin accounting, mirroring ClearMotMetrics._compute_mota_and_motp."""
    truth_at = defaultdict(dict)
    for p in truths:
        for s in p.states:
            truth_at[s.timestamp][p.id] = s
    track_at = defaultdict(dict)
    for t in tracks:
        for s in t.states:
            track_at[s.timestamp][t.id] = s

    matches_at = defaultdict(set)
    for ts in sorted(set(truth_at) | set(track_at)):
        for a in assoc_set.associations_at_timestamp(ts):
            objs = list(a.objects)
            truth_ids = {p.id for p in truths}
            tr_id = next((o.id for o in objs if o.id in truth_ids), None)
            tk_id = next((o.id for o in objs if o.id not in truth_ids), None)
            if tr_id is not None and tk_id is not None:
                matches_at[ts].add((tr_id, tk_id))

    stamps = sorted(set(truth_at) | set(track_at))
    misses = fps = mismatches = n_assoc = 0
    err = 0.0
    for i, ts in enumerate(stamps):
        cur = matches_at[ts]
        mt = {m[0] for m in cur}
        mk = {m[1] for m in cur}
        for tr_id, tk_id in cur:
            a = truth_at[ts][tr_id].state_vector
            b = track_at[ts][tk_id].state_vector
            err += float(np.hypot(a[0] - b[0], a[1] - b[1]))
        n_assoc += len(cur)
        misses += len(set(truth_at[ts]) - mt)
        fps += len(set(track_at[ts]) - mk)
        if i:
            prev = matches_at[stamps[i - 1]]
            pm = {m[0] for m in prev}
            for tr_id in pm & mt:
                if next(m[1] for m in prev if m[0] == tr_id) != \
                   next(m[1] for m in cur if m[0] == tr_id):
                    mismatches += 1
    n_gt = sum(len(p.states) for p in truths)
    return dict(fn=misses, fp=fps, idsw=mismatches, gt=n_gt,
                dist_sum=err, n_matched=n_assoc)


def score_segment(tracks, truths, ad, cross_check=True):
    assoc = ClearMotAssociator(association_threshold=ad, measure=Euclidean())
    aset = assoc.associate_tracks(tracks, truths)
    c = clearmot_counts(tracks, truths, aset)

    if cross_check and c["gt"] and c["n_matched"]:
        gen = ClearMotMetrics(generator_name="x", tracks_key="tracks",
                              truths_key="groundtruth_paths", distance_measure=Euclidean())
        mgr = MultiManager([gen], associator=assoc)
        mgr.add_data({"tracks": tracks, "groundtruth_paths": truths})
        try:
            got = {m.title: m.value for m in mgr.generate_metrics()["x"].values()} \
                if isinstance(mgr.generate_metrics()["x"], dict) else \
                {m.title: m.value for m in mgr.generate_metrics()["x"]}
            mine_motp = c["dist_sum"] / c["n_matched"]
            mine_mota = 1 - (c["fn"] + c["fp"] + c["idsw"]) / c["gt"]
            c["xcheck_motp_delta"] = abs(got.get("MOTP", np.nan) - mine_motp)
            c["xcheck_mota_delta"] = abs(got.get("MOTA", np.nan) - mine_mota)
        except Exception as e:
            c["xcheck_error"] = repr(e)[:120]
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--no-cross-check", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = cfg["paths"]
    decimate = int(cfg["sampling"]["decimate"])

    if not os.path.exists(p["gt"]):
        sys.exit(f"missing {p['gt']} -- run mot_benchmark_frog/frog_gt.py first")
    gt = np.load(p["gt"], allow_pickle=True)
    dumps = sorted(glob.glob(os.path.join(p["out_dir"], f"tracks_{p['bag']}_*.npz")))
    if not dumps:
        sys.exit("no track dumps found -- run replay.py first")

    print(f"{'variant':>22} {'tracker':>8} {'pipeline':>10} {'conf':>5} {'ad':>4} "
          f"{'MOTA':>8} {'MOTP(m)':>8} {'FN':>7} {'FP':>7} {'IDSW':>5} {'GT':>7}")
    print("-" * 104)
    results = []
    json_rows = []
    for dump in dumps:
        tr = np.load(dump, allow_pickle=True)
        mode, conf = str(tr["pipeline"]), float(tr["conf_thresh"])
        variant = str(tr["variant"]) if "variant" in tr.files else "?"
        trk = str(tr["tracker"]) if "tracker" in tr.files else "kf"
        for ad in cfg["scoring"]["association_distances"]:
            tot = defaultdict(float)
            worst = 0.0
            covered = set(tr["segments_run"].tolist()) if "segments_run" in tr.files \
                else set(np.unique(gt["segment"]).tolist())
            for seg in sorted(covered):
                rows = np.nonzero(gt["segment"] == seg)[0][::decimate]
                truths = build_truths(gt, rows)
                tracks = build_tracks(tr, int(seg))
                if not truths:
                    continue
                c = score_segment(tracks, truths, ad, not args.no_cross_check)
                for k in ("fn", "fp", "idsw", "gt", "dist_sum", "n_matched"):
                    tot[k] += c[k]
                worst = max(worst, c.get("xcheck_mota_delta", 0.0) or 0.0)
            mota = 1 - (tot["fn"] + tot["fp"] + tot["idsw"]) / tot["gt"] if tot["gt"] else float("nan")
            motp = tot["dist_sum"] / tot["n_matched"] if tot["n_matched"] else float("nan")
            print(f"{variant:>22} {trk:>8} {mode:>10} {conf:>5} {ad:>4} {mota:>8.4f} {motp:>8.4f} "
                  f"{int(tot['fn']):>7} {int(tot['fp']):>7} {int(tot['idsw']):>5} {int(tot['gt']):>7}")
            results.append((f'{variant}/{trk}', mode, conf, ad, mota, motp, tot))
            json_rows.append(dict(
                variant=variant, tracker=trk, pipeline=mode, conf=conf, ad=ad,
                mota=mota, motp=motp, fn=int(tot["fn"]), fp=int(tot["fp"]),
                idsw=int(tot["idsw"]), gt=int(tot["gt"]),
                xcheck_max_mota_delta=worst if worst > 1e-9 else 0.0,
            ))
            if worst > 1e-9:
                print(f"           ^ cross-check max |ΔMOTA| vs stonesoup = {worst:.2e}")

    seqs = {(v, c, a): (m1, m2) for v, md, c, a, m1, m2, _ in results if md == "sequential"}
    pips = {(v, c, a): (m1, m2) for v, md, c, a, m1, m2, _ in results if md == "pipelined"}
    both = set(seqs) & set(pips)
    seq_vs_pipelined_max_delta = None
    if both:
        d = max(max(abs(seqs[k][0] - pips[k][0]), abs(seqs[k][1] - pips[k][1])) for k in both)
        seq_vs_pipelined_max_delta = d
        print(f"\nsequential vs pipelined: max |Δ| across {len(both)} settings = {d:.3e}")
        print("  (expected 0 -- the pipelined node never drops a frame and its single"
              "\n   consumer preserves order and dt, so the tracking maths is identical)")

    os.makedirs(p["results_dir"], exist_ok=True)
    stamp = dt.datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(p["results_dir"], f"{p['bag']}_20Hz_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(dict(
            timestamp=dt.datetime.now(UTC).isoformat(),
            bag=p["bag"],
            decimate=decimate,
            cross_check=not args.no_cross_check,
            dumps_scored=[os.path.basename(d) for d in dumps],
            rows=json_rows,
            sequential_vs_pipelined_max_delta=seq_vs_pipelined_max_delta,
        ), f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
