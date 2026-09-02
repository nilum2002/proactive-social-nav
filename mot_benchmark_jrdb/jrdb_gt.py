"""Build ground-truth tracks (with REAL identities) for the JRDB MOT benchmark.

Unlike mot_benchmark_drowv2/pseudo_gt.py, no linking heuristic is needed:
JRDB's 3D box labels carry a persistent `label_id` per sequence already. This
script just flattens TEST_SEQUENCES' labeled frames into one row-indexed
table, namespacing `label_id` by sequence (it is only unique *within* one
sequence -- "pedestrian:0" in two different sequences is two different
people), mirroring mot_benchmark_frog/frog_gt.py's `idp` chunk-namespacing.

Each JRDB sequence is treated as one "segment" -- the same role FROG's 55
temporally-contiguous segments play: replay.py and score.py reset/reset-score
per segment, never carrying state across a sequence boundary.
"""
import argparse
import os
import sys

import numpy as np

from jrdb_common import load_config, list_sequence_frames, DEFAULT_CONFIG, TEST_SEQUENCES


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = cfg["paths"]
    os.makedirs(p["out_dir"], exist_ok=True)

    train_dir = os.path.join(p["data_dir"], "train_dataset")
    if not os.path.isdir(train_dir):
        sys.exit(
            f"missing {train_dir} -- see config.yaml's PREREQUISITES header "
            "(download JRDB + log in, then run dr_spaam/bin/setup_jrdb_dataset.py)"
        )

    segment, timestamp, laser_path = [], [], []
    gt_scan, gt_id, gt_xy = [], [], []
    id_names = []
    id_index = {}
    row = 0

    for seg_idx, seq in enumerate(TEST_SEQUENCES):
        frames = list_sequence_frames(p["data_dir"], seq)
        if not frames:
            print(f"  {seq:<45} 0 labeled frames -- skipped (missing labels/lasers?)")
            continue
        for fr in frames:
            segment.append(seg_idx)
            timestamp.append(fr["timestamp"])
            laser_path.append(fr["laser_path"])
            for label_id, x, y in fr["anns"]:
                key = f"{seq}#{label_id}"
                if key not in id_index:
                    id_index[key] = len(id_names)
                    id_names.append(key)
                gt_scan.append(row)
                gt_id.append(id_index[key])
                gt_xy.append((x, y))
            row += 1
        n_ann = sum(len(fr["anns"]) for fr in frames)
        print(f"  {seq:<45} {len(frames):>5} labeled frames, {n_ann:>6} person-observations")

    if row == 0:
        sys.exit("no labeled frames found across TEST_SEQUENCES -- check data_dir/setup")

    out = os.path.join(p["out_dir"], "jrdb_test_gt.npz")
    np.savez_compressed(
        out,
        segment=np.array(segment, dtype=np.int64),
        timestamp=np.array(timestamp, dtype=np.float64),
        laser_path=np.array(laser_path),
        gt_scan=np.array(gt_scan, dtype=np.int64),
        gt_id=np.array(gt_id, dtype=np.int64),
        gt_xy=np.array(gt_xy, dtype=np.float64).reshape(-1, 2),
        gt_id_names=np.array(id_names),
        sequence_names=np.array(TEST_SEQUENCES),
    )
    print(f"\nwrote {out}  ({row} rows across {len(TEST_SEQUENCES)} sequences, "
          f"{len(id_names)} ground-truth identities, {len(gt_scan)} person-observations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
