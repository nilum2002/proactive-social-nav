"""Score replayed DROWv2 tracks against pseudo ground truth with CLEAR MOT
(MOTA / MOTP). SECONDARY, approximate benchmark -- see pseudo_gt.py's
docstring first. `drow_ap.py` is the primary, literature-comparable-in-shape
benchmark in this directory.

Association is done with Stone Soup's `ClearMotAssociator`, which is the
Bernardin et al. (2008) rule: a track keeps its association with a truth from
the previous timestep even if a closer track appears, and anything left over is
matched one-to-one by Munkres inside the distance gate.

The accounting is done here rather than by `ClearMotMetrics` for one reason:
that class computes num_misses / num_false_positives / num_miss_matches /
number_of_gt_states and then returns only (MOTP, MOTA), discarding the
components. Two things need them.

  * Reporting. FN / FP / IDSW are what make a MOTA number diagnosable.
  * Pooling. The DROWv2 test split is several disjoint sequences, so scores
    must be combined as 1 - (sum FN + sum FP + sum IDSW) / sum GT and
    sum(distance) / sum(matches). Averaging per-sequence MOTA values would
    weight a short sequence the same as a long one.

To keep that honest, every segment is *also* scored with Stone Soup's own
ClearMotMetrics and the two are cross-checked before pooling (ported verbatim
from mot_benchmark/score.py, including the Stone Soup 1.9.1 zero-length
TimeRange patch below and the deterministic-ordering fix for build_truths/
build_tracks -- both apply here for the same reasons, unrelated to which
dataset the tracks/truths come from).

Unlike FROG, "ground truth" here is `pseudo_gt.py`'s synthesized tracklets,
not verified identity -- IDSW in particular reflects that linking heuristic,
not confirmed real id switches. Treat FN/FP as far more trustworthy than IDSW.

No odometry: same as FROG, tracking and ground truth are both in the sensor
frame. DROW sequences may or may not have significant robot motion depending
on the run; this was not characterized before writing this benchmark.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from drow_common import load_config, DEFAULT_CONFIG

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

# Stone Soup 1.9.1 bug: ClearMotAssociator._create_associations_from_matches_over_time
# builds a TimeRange(start, end) per contiguous run of a (track, truth) match, but for a
# match that exists at exactly one timestep, start == end, and Interval.__init__ demands
# start < end -- so a perfectly normal single-frame association crashes the whole run.
# Nudge a degenerate datetime range's end forward by 1us (below any plausible scan
# spacing, so it can never reach the next real timestamp) instead of raising.
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
    """Epoch seconds -> datetime. Both sides must key on identical objects:
    Stone Soup matches timestamps by equality, not by tolerance."""
    return dt.datetime.fromtimestamp(round(float(x), 6), tz=UTC)


def build_truths(gt, rows):
    """One GroundTruthPath per pseudo-tracklet id present in these rows.

    Returns a list sorted by id, not a set -- see mot_benchmark/score.py's
    build_truths docstring for why (identity-hash set ordering makes results
    non-reproducible across process runs otherwise).
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
    """One Track per tracker id within this segment (segment = DROW sequence).

    Returns a list sorted by id, not a set -- see build_truths.
    """
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
    gt_path = os.path.join(p["out_dir"], f"drowv2_{p['split']}_pseudo_gt.npz")
    if not os.path.exists(gt_path):
        sys.exit(f"missing {gt_path} -- run pseudo_gt.py first")
    gt = np.load(gt_path, allow_pickle=True)
    dumps = sorted(glob.glob(os.path.join(p["out_dir"], f"drowv2_{p['split']}_tracks_*.npz")))
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
                rows = np.nonzero(gt["segment"] == seg)[0]
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
    out_path = os.path.join(p["results_dir"], f"mot_{p['split']}_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(dict(
            timestamp=dt.datetime.now(UTC).isoformat(),
            split=p["split"],
            cross_check=not args.no_cross_check,
            dumps_scored=[os.path.basename(d) for d in dumps],
            rows=json_rows,
            sequential_vs_pipelined_max_delta=seq_vs_pipelined_max_delta,
            caveat="pseudo ground truth (see pseudo_gt.py) -- IDSW is a linking-heuristic "
                   "artifact, not verified identity; FN/FP are more trustworthy",
        ), f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
