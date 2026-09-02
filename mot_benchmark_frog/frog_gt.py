"""Build ground-truth tracks (with identities) for the FROG dataset.

The released HDF5 files carry per-scan person annotations but no identity: the
`circles` array is (M, 6) = x, y, radius, distance, angle, angular-radius, and
`convert_frog_circles.py` in the official benchmark reads only x/y/r/type,
dropping the labelling tool's `idp` field on the way in.

The raw CSVs published alongside the HDF5 (frog-raw-circles.zip) keep it:

    id,timestamp,circles
    1453,1398781620.9157312,[{'idp': 3, 'x': 9.94, 'y': -0.96, 'r': 0.4, 'type': 1}]

`idp` is the person tracklet identifier, temporally consistent within a
tracklet. The paper is explicit that these are *partial* tracklets -- one person
may span several ids across occlusions -- and that no existing detection work
uses them. That fragmentation is left in place here; see the README note in
score.py for why it costs almost nothing under CLEAR MOT.

This module reconciles the two sources and emits per-scan ground-truth tracks
aligned to HDF5 row order.
"""
import argparse
import glob
import json
import re
import os
import sys

import numpy as np
import yaml

# Constants replicated from the official convert_frog_circles.py. A circle that
# this filter would have dropped is absent from the HDF5, so keeping it here
# would invent ground truth the benchmark never had -- every such person would
# score as a permanent miss.
SCAN_NEAR = 0.2
SCAN_FAR = 10.0
SCAN_ANGMIN = -np.pi / 2
SCAN_ANGMAX = +np.pi / 2


DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path=DEFAULT_CONFIG):
    """Load the config and make every path absolute.

    `repo_root` is resolved relative to the *config file*, not the process cwd,
    so these scripts behave the same whether they are run from the repository
    root or from inside mot_benchmark/.
    """
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(path)
    root = os.path.abspath(os.path.join(cfg_dir, cfg["paths"].get("repo_root", "..")))
    cfg["paths"]["repo_root"] = root
    for k, v in cfg["paths"].items():
        if k not in ("repo_root", "bag"):
            cfg["paths"][k] = v if os.path.isabs(v) else os.path.join(root, v)
    return cfg


def _parse_circles(raw_json):
    """Parse one CSV `circles` cell, applying the official bounds filter.

    Returns rows of (idp, x, y, r, dist, angle) already in the converter's sort
    order, so the result can be compared row-for-row against the HDF5.
    """
    # The tool writes Python repr, not JSON: single quotes.
    objs = json.loads(raw_json.replace("'", '"'))
    out = []
    for c in objs:
        if c.get("type", 1) != 1:
            continue  # non-people classes (strollers, walking aids) are excluded
        x, y, r = float(c["x"]), float(c["y"]), float(c["r"])
        dist = float(np.hypot(x, y))
        angle = float(np.arctan2(y, x))
        if dist < SCAN_NEAR or dist >= SCAN_FAR:
            continue
        if angle < SCAN_ANGMIN or angle >= SCAN_ANGMAX:
            continue
        out.append((int(c["idp"]), x, y, r, dist, angle))
    # convert_frog_circles.py: temp.sort(key=lambda c: (-c[3], c[4])) on
    # (x, y, r, dist, angle, angr) -- i.e. back-to-front, then CCW.
    out.sort(key=lambda t: (-t[4], t[5]))
    return out


def load_raw_circles(circles_dir, bag):
    """Read every chunk CSV for one bag into per-scan annotations.

    Two hazards handled here:

    * `idp` is only unique *within* a chunk file. In the 16-41 test set, chunk
      0-40472 shares 139 idp values with chunk 40472-52730. Identities are
      therefore namespaced as "<chunk>#<idp>"; keying on the bare integer would
      weld unrelated people into one ground-truth track and fabricate ID
      switches that never happened.
    * Chunk boundaries are inclusive on both sides, so scans 40472 / 52730 /
      62302 appear in two files each. Their two copies are not always equal:
      in the 16-41 test set, chunk 0-40472 annotates one person (idp 500) on
      scan 40472 while chunk 40472-52730 gives it `[]`. The published HDF5
      sides with the *later* chunk -- that scan is absent from it entirely --
      so the later chunk wins here too. Keeping the earlier copy leaves one
      extra scan and shifts every subsequent row by one against the HDF5.

    Chunks are ordered by their numeric start index rather than by filename:
    lexicographic order happens to agree for the published bags, but only by
    luck of the naming.
    """
    files = glob.glob(os.path.join(circles_dir, f"bag_{bag}_*_circles.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs matching bag_{bag}_*_circles.csv in {circles_dir}")

    def _start(path):
        m = re.search(rf"bag_{re.escape(bag)}_(\d+)[-_]", os.path.basename(path))
        return int(m.group(1)) if m else 0
    files = sorted(files, key=_start)

    per_scan = {}   # raw scan id -> (timestamp, [(gt_key, x, y, r), ...])
    for path in files:
        chunk = os.path.basename(path).replace("_circles.csv", "")
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    continue  # header: id,timestamp,circles
                sid_s, ts_s, cj = line.rstrip("\r\n").split(",", 2)
                sid = int(sid_s)
                # No `continue` on a repeat: later chunks deliberately overwrite
                # earlier ones on shared boundary scans (see the docstring).
                people = [(f"{chunk}#{idp}", x, y, r)
                          for idp, x, y, r, _, _ in _parse_circles(cj)]
                per_scan[sid] = (float(ts_s), people)

    scan_ids = np.array(sorted(per_scan), dtype=np.int64)
    ts = np.array([per_scan[s][0] for s in scan_ids], dtype=np.float64)
    people = [per_scan[s][1] for s in scan_ids]
    return scan_ids, ts, people, [os.path.basename(f) for f in files]


def keep_mask(people):
    """The HDF5 test/train files are exported with strip_empty=True, keeping
    only scans that hold at least one person."""
    return np.array([len(p) > 0 for p in people], dtype=bool)


def find_segments(scan_ids):
    """Split the kept scans into temporally contiguous runs.

    strip_empty leaves holes wherever nobody was annotated -- in the 16-41 test
    set, 55 segments separated by gaps of up to 81 s. Tracking straight through
    a hole would carry stale tracks across a minute of missing time and
    manufacture false positives and ID switches, so every consumer of this file
    must reset its tracker at a segment boundary.
    """
    if len(scan_ids) == 0:
        return np.zeros(0, dtype=np.int64)
    brk = np.nonzero(np.diff(scan_ids) != 1)[0]
    seg = np.zeros(len(scan_ids), dtype=np.int64)
    seg[brk + 1] = 1
    return np.cumsum(seg)


def join_to_h5(h5_path, kept_ids, kept_ts, kept_people):
    """Verify the kept-scan sequence lines up with the HDF5 row order.

    The join is by *rank*, not by timestamp: the HDF5 is exactly the ordered
    subsequence of scans with circle_num > 0, and epoch timestamps are fragile
    under any float32 round-trip (1.4e9 is well past float32's ~7 significant
    digits). Timestamps and per-scan person counts are then used to prove the
    rank alignment is right rather than to establish it.
    """
    import h5py

    report = {}
    with h5py.File(h5_path, "r") as f:
        h5_ts = f["timestamps"][:]
        circle_num = f["circle_num"][:]
        circles = f["circles"][:]
        circle_idx = f["circle_idx"][:]

    report["h5_scans"] = len(h5_ts)
    report["csv_kept"] = len(kept_ids)
    report["count_match"] = len(h5_ts) == len(kept_ids)
    n = min(len(h5_ts), len(kept_ids))

    dt = np.abs(np.asarray(h5_ts[:n], dtype=np.float64) - kept_ts[:n])
    report["ts_max_abs_diff_s"] = float(dt.max()) if n else float("nan")
    report["ts_within_1ms"] = float((dt < 1e-3).mean()) if n else float("nan")

    csv_counts = np.array([len(p) for p in kept_people[:n]])
    report["count_agree_frac"] = float((csv_counts == circle_num[:n]).mean()) if n else float("nan")
    bad = np.nonzero(csv_counts != circle_num[:n])[0]
    report["first_count_mismatches"] = bad[:10].tolist()

    # Position agreement on a sample: same people, same converter sort order.
    rng = np.random.default_rng(0)
    probe = rng.choice(n, size=min(500, n), replace=False) if n else []
    worst = 0.0
    for i in probe:
        k, m = int(circle_idx[i]), int(circle_num[i])
        h5_xy = circles[k:k + m, :2]
        csv_xy = np.array([[x, y] for _, x, y, _ in kept_people[i]], dtype=np.float64)
        if h5_xy.shape != csv_xy.shape:
            worst = float("inf")
            break
        worst = max(worst, float(np.abs(h5_xy - csv_xy).max()))
    report["max_xy_diff_on_sample"] = worst
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--skip-h5", action="store_true",
                    help="build GT from the CSVs alone; skip the HDF5 cross-check")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = cfg["paths"]
    os.makedirs(p["out_dir"], exist_ok=True)

    scan_ids, ts, people, files = load_raw_circles(p["circles_dir"], p["bag"])
    print(f"chunk files      : {len(files)}")
    print(f"raw scans        : {len(scan_ids)} (after de-duplicating chunk boundaries)")

    m = keep_mask(people)
    kept_ids, kept_ts = scan_ids[m], ts[m]
    kept_people = [pp for pp, keep in zip(people, m) if keep]
    print(f"scans with people: {m.sum()}  (these are what the HDF5 keeps)")

    seg = find_segments(kept_ids)
    n_seg = seg.max() + 1 if len(seg) else 0
    lens = np.bincount(seg) if len(seg) else np.array([])
    print(f"segments         : {n_seg}  (median {np.median(lens):.0f} scans, max {lens.max()} )")

    ids = sorted({gid for pp in kept_people for gid, *_ in pp})
    id_index = {g: i for i, g in enumerate(ids)}
    print(f"ground-truth ids : {len(ids)} tracklets")

    gt_scan, gt_id, gt_xy, gt_r = [], [], [], []
    for row, pp in enumerate(kept_people):
        for gid, x, y, r in pp:
            gt_scan.append(row); gt_id.append(id_index[gid]); gt_xy.append((x, y)); gt_r.append(r)

    out = os.path.join(p["out_dir"], f"frog_{p['bag']}_gt.npz")
    np.savez_compressed(
        out,
        raw_scan_id=kept_ids, timestamp=kept_ts, segment=seg,
        gt_scan=np.array(gt_scan, dtype=np.int64),
        gt_id=np.array(gt_id, dtype=np.int64),
        gt_xy=np.array(gt_xy, dtype=np.float64).reshape(-1, 2),
        gt_r=np.array(gt_r, dtype=np.float64),
        gt_id_names=np.array(ids),
    )
    print(f"\nwrote {out}  ({len(gt_scan)} person-observations)")

    if args.skip_h5 or not os.path.exists(p["h5"]):
        print(f"\nHDF5 not checked ({'--skip-h5' if args.skip_h5 else p['h5'] + ' not found'}).")
        print("Download frog_16-41_test.h5 and re-run to validate the alignment.")
        return 0

    print(f"\ncross-checking against {os.path.basename(p['h5'])} ...")
    rep = join_to_h5(p["h5"], kept_ids, kept_ts, kept_people)
    for k, v in rep.items():
        print(f"  {k:<26}: {v}")
    ok = rep["count_match"] and rep["count_agree_frac"] == 1.0 and rep["max_xy_diff_on_sample"] < 1e-3
    print("\nALIGNMENT OK" if ok else "\nALIGNMENT MISMATCH -- do not trust downstream scores")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
