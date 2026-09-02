"""Synthesize pseudo-tracklets for DROWv2's test split, for the SECONDARY,
approximate MOT benchmark only (pseudo_gt.py -> replay.py -> score.py).

DROWv2's public `.wp` annotations are a flat per-scan list of (r, phi) points
with no persistent person identity -- unlike FROG, there is no `idp` field or
raw labelling-tool export to recover one from. This script fakes one: within
each sequence, it greedily nearest-neighbor-links GT points across
consecutive ANNOTATED frames (Hungarian assignment, gated by
`pseudo_gt.max_link_dist` and `max_link_gap_s`), and treats each resulting
chain as one tracklet.

Treat every MOTA/MOTP number downstream of this file as an approximation, not
ground truth: an unlucky nearest-neighbor pick fabricates an id switch that
never happened, and a lucky one hides a real one. IDSW in particular reflects
this linking heuristic at least as much as it reflects the tracker under
test. FN/FP are more trustworthy, since those only need "is there a truth
point/track point here", not identity.

Also note DROW's annotations are sparse (~1 in 20 scans, see drow_common.py).
Scoring only makes sense at those annotated instants -- replay.py restricts
its dumped track positions to the same frames for exactly this reason.
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from drow_common import load_config, DEFAULT_CONFIG, list_sequences, Sequence


def link_sequence(seq, max_link_dist, max_link_gap_s):
    """Returns (gt_scan, gt_id, gt_xy) arrays: one (scan_row, id, xy) triple
    per annotated person point, `scan_row` indexing this sequence's own
    annotated-frame list (0..len(seq.ann_ns)-1)."""
    gt_scan, gt_id, gt_xy = [], [], []
    active = {}  # track_id -> (x, y, t)
    next_id = 0

    for k in range(len(seq.ann_ns)):
        t = seq.t[seq.ann_scan_idx[k]]
        wp = seq.wp[k]
        pts = np.stack(_rphi_to_xy(wp[:, 0], wp[:, 1]), axis=1) if len(wp) else \
            np.zeros((0, 2), dtype=np.float32)

        if active and (t - next(iter(active.values()))[2]) > max_link_gap_s:
            active = {}

        matched_new = set()
        if active and len(pts) > 0:
            ids = list(active.keys())
            prev_xy = np.array([active[i][:2] for i in ids])
            cost = np.linalg.norm(prev_xy[:, None, :] - pts[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= max_link_dist:
                    tid = ids[r]
                    active[tid] = (pts[c, 0], pts[c, 1], t)
                    gt_scan.append(k)
                    gt_id.append(tid)
                    gt_xy.append(pts[c])
                    matched_new.add(c)

        # unmatched points start new tracklets
        for c in range(len(pts)):
            if c in matched_new:
                continue
            tid = next_id
            next_id += 1
            active[tid] = (pts[c, 0], pts[c, 1], t)
            gt_scan.append(k)
            gt_id.append(tid)
            gt_xy.append(pts[c])

        # drop tracks that found no match this frame -- they cannot be
        # continued later without pretending across a silent gap
        seen_ids = {gt_id[i] for i in range(len(gt_scan)) if gt_scan[i] == k}
        active = {i: v for i, v in active.items() if i in seen_ids}

    return (np.array(gt_scan, dtype=np.int64), np.array(gt_id, dtype=np.int64),
            np.array(gt_xy, dtype=np.float32).reshape(-1, 2))


def _rphi_to_xy(r, phi):
    return r * np.cos(phi), r * np.sin(phi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = cfg["paths"]
    pg = cfg["pseudo_gt"]

    seqs = list_sequences(p["data_dir"], p["split"])
    if not seqs:
        sys.exit(f"no sequences found under {p['data_dir']}/{p['split']}")

    all_scan, all_id, all_xy, all_ts, all_seg = [], [], [], [], []
    row_ts, row_seg = [], []  # one entry per annotated frame, across all sequences
    n_tracklets_total = 0

    for seg, seq_path in enumerate(seqs):
        seq = Sequence(seq_path)
        gt_scan, gt_id, gt_xy = link_sequence(seq, pg["max_link_dist"], pg["max_link_gap_s"])

        row_base = len(row_ts)
        for k in range(len(seq.ann_ns)):
            row_ts.append(float(seq.t[seq.ann_scan_idx[k]]))
            row_seg.append(seg)

        all_scan.append(gt_scan + row_base)
        all_id.append(gt_id)
        all_xy.append(gt_xy)
        n_tracklets = len(set(gt_id.tolist()))
        n_tracklets_total += n_tracklets
        print(f"{seq.name}: {len(seq.ann_ns)} annotated frames, "
              f"{len(gt_id)} person-observations, {n_tracklets} pseudo-tracklets")

    gt_scan = np.concatenate(all_scan) if all_scan else np.zeros(0, dtype=np.int64)
    gt_id = np.concatenate(all_id) if all_id else np.zeros(0, dtype=np.int64)
    gt_xy = np.concatenate(all_xy, axis=0) if all_xy else np.zeros((0, 2), dtype=np.float32)
    timestamp = np.array(row_ts, dtype=np.float64)
    segment = np.array(row_seg, dtype=np.int64)

    os.makedirs(p["out_dir"], exist_ok=True)
    out_path = os.path.join(p["out_dir"], f"drowv2_{p['split']}_pseudo_gt.npz")
    np.savez_compressed(
        out_path, gt_scan=gt_scan, gt_id=gt_id, gt_xy=gt_xy,
        timestamp=timestamp, segment=segment,
        sequences=np.array(seqs, dtype=object),
    )
    print(f"\n{len(gt_id)} person-observations, {n_tracklets_total} pseudo-tracklets total")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
