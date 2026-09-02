"""AP / precision-recall for DR-SPAAM(T=1) vs DR-SPAAM(T=5) on the FROG test
split -- the mAP / mPeak-F1 / mEER table format FROG's own leaderboard uses
(mean over d=0.5m and d=0.3m association radii, plus each radius broken out).

Same windowed (T-scan, gate-reset-per-sample) evaluation protocol as
mot_benchmark_drowv2/drow_ap.py -- see that file's docstring for why this is
the protocol a checkpoint's own "T=N" training regime was tuned against, not
the streaming persistent-memory mode mot_benchmark_frog/replay.py uses for
deployment-style MOT scoring. `num_scans` per checkpoint is set in
config.yaml's `checkpoints:` list to match each one's own T.

Ground truth is not rebuilt here: it reuses mot_benchmark_frog/frog_gt.py's
already cross-checked frog_16-41_gt.npz (run that script first if missing)
and the matching frog_dataset/frog_16-41_test.h5 scans, grouped into FROG's
existing 55 temporally-contiguous segments so a T-scan window never crosses a
segment gap.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np
import torch

from frog_common import load_config, DEFAULT_CONFIG, build_scan_phi, preprocess, GroundTruth

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dr_spaam"))

from dr_spaam.model.drow_net import DrowNet
from dr_spaam.model.dr_spaam import DrSpaam
import dr_spaam.utils.utils as u
import dr_spaam.utils.precision_recall as pru


def load_model(ckpt_file, model_name, gpu):
    """Mirrors dr_spaam.detector.Detector.__init__ / drow_ap.py::load_model --
    same hardcoded cutout/gate hyperparams regardless of laser geometry."""
    if model_name == "DROW3":
        model = DrowNet(dropout=0.5, cls_loss=None, mixup_alpha=0.0, mixup_w=0.0)
    elif model_name == "DR-SPAAM":
        model = DrSpaam(
            dropout=0.5, num_pts=56, embedding_length=128, alpha=0.5,
            window_size=17, panoramic_scan=False, cls_loss=None,
            mixup_alpha=0.0, mixup_w=0.0,
        )
    else:
        raise NotImplementedError(model_name)

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if gpu:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    return model


def predict(model, use_dr_spaam, scans, scan_phi, cutout_kwargs, stride, gpu):
    """One windowed forward pass -- `scans` is (T, num_pts), oldest first."""
    ct = u.scans_to_cutout(scans, scan_phi, stride=stride, **cutout_kwargs)
    ct = torch.from_numpy(ct).float().unsqueeze(0)
    if gpu:
        ct = ct.cuda()
    with torch.no_grad():
        if use_dr_spaam:
            pred_cls, pred_reg, _ = model(ct, inference=False)
        else:
            pred_cls, pred_reg = model(ct)
    pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()[:, 0]
    pred_reg = pred_reg[0].data.cpu().numpy()
    return u.nms_predicted_center(scans[-1, ::stride], scan_phi[::stride], pred_cls, pred_reg)


def run_checkpoint(ck, gt, cfg, ap_root, max_segments=None):
    p, fr = cfg["paths"], cfg["frog"]
    gpu = bool(cfg["detector"]["gpu"])
    stride = cfg["detector"]["scan_stride"]
    scan_phi = build_scan_phi(fr["num_pts"], fr["fov_deg"])

    weight_file = ck["weight_file"]
    if not os.path.isabs(weight_file):
        weight_file = os.path.join(p["repo_root"], weight_file)
    model = load_model(weight_file, ck["model"], gpu)
    use_dr_spaam = ck["model"] == "DR-SPAAM"
    num_scans = ck["num_scans"]

    ap_dir = os.path.join(ap_root, ck["name"].replace("/", "_"))
    shutil.rmtree(ap_dir, ignore_errors=True)
    os.makedirs(os.path.join(ap_dir, "detections"), exist_ok=True)
    os.makedirs(os.path.join(ap_dir, "groundtruth"), exist_ok=True)

    seg_ids = gt.seg_ids[:max_segments] if max_segments else gt.seg_ids
    for seg in seg_ids:
        rows = gt.segment_rows(int(seg))
        scans = gt.scans_all[rows]
        seq_name = f"seg{int(seg):03d}"
        os.makedirs(os.path.join(ap_dir, "detections", seq_name), exist_ok=True)
        os.makedirs(os.path.join(ap_dir, "groundtruth", seq_name), exist_ok=True)

        for local_idx, row in enumerate(rows):
            window = GroundTruth.window(scans, local_idx, num_scans)
            window = np.stack([preprocess(s, fr["range_min"], fr["range_max"]) for s in window])
            dets_xy, dets_cls, _ = predict(
                model, use_dr_spaam, window, scan_phi, cfg["cutout_kwargs"], stride, gpu)
            frame_id = f"{local_idx:06d}"

            det_str = pru.drow_detection_to_kitti_string(dets_xy, dets_cls, None)
            with open(os.path.join(ap_dir, "detections", seq_name, f"{frame_id}.txt"), "w") as f:
                f.write(det_str)

            gts_xy = gt.points_for_row(int(row))
            occluded = np.zeros(len(gts_xy), dtype=int)  # FROG's circles carry no occlusion field
            gt_str = pru.drow_detection_to_kitti_string(gts_xy, None, occluded) if len(gts_xy) else ""
            with open(os.path.join(ap_dir, "groundtruth", seq_name, f"{frame_id}.txt"), "w") as f:
                f.write(gt_str)

    sequences, res_03, res_05 = pru.evaluate_drow(ap_dir, verbose=False)
    # evaluate_drow appends an "all" aggregate as the last entry iff there was
    # more than one sequence -- true here (55 FROG segments).
    idx = sequences.index("all") if "all" in sequences else -1
    r3, r5 = res_03[idx], res_05[idx]
    row = dict(
        name=ck["name"], weight_file=os.path.relpath(weight_file, p["repo_root"]),
        num_scans=num_scans,
        ap_0_5=float(r5["ap"]), peak_f1_0_5=float(r5["peak_f1"]), eer_0_5=float(r5["eer"]),
        ap_0_3=float(r3["ap"]), peak_f1_0_3=float(r3["peak_f1"]), eer_0_3=float(r3["eer"]),
    )
    row["map"] = (row["ap_0_5"] + row["ap_0_3"]) / 2
    row["mpeak_f1"] = (row["peak_f1_0_5"] + row["peak_f1_0_3"]) / 2
    row["meer"] = (row["eer_0_5"] + row["eer_0_3"]) / 2
    return row


def print_table(rows):
    hdr = (f"{'':<16} {'mAP':>6} {'mPeakF1':>8} {'mEER':>6}  |  "
           f"{'AP@.5':>6} {'PeakF1@.5':>10} {'EER@.5':>7}  |  "
           f"{'AP@.3':>6} {'PeakF1@.3':>10} {'EER@.3':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<16} {r['map']:>6.3f} {r['mpeak_f1']:>8.3f} {r['meer']:>6.3f}  |  "
              f"{r['ap_0_5']:>6.3f} {r['peak_f1_0_5']:>10.3f} {r['eer_0_5']:>7.3f}  |  "
              f"{r['ap_0_3']:>6.3f} {r['peak_f1_0_3']:>10.3f} {r['eer_0_3']:>7.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", help="run only the named checkpoint (config.yaml checkpoints[].name)")
    ap.add_argument("--max-segments", type=int, default=None, help="limit segments (smoke tests)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = cfg["paths"]

    if not os.path.exists(p["gt"]):
        sys.exit(f"missing {p['gt']} -- run mot_benchmark_frog/frog_gt.py first")
    gt = GroundTruth(p["gt"], p["h5"])

    checkpoints = cfg["checkpoints"]
    if args.checkpoint:
        checkpoints = [c for c in checkpoints if c["name"] == args.checkpoint]
        if not checkpoints:
            sys.exit(f"no checkpoint named '{args.checkpoint}' in config.yaml")

    rows = []
    for ck in checkpoints:
        print(f"=== {ck['name']}  ({ck['weight_file']}, T={ck['num_scans']}) ===")
        row = run_checkpoint(ck, gt, cfg, p["ap_dir"], args.max_segments)
        rows.append(row)
        print_table([row])
        print()

    if len(rows) > 1:
        print("=== summary =====================================================")
        print_table(rows)

    os.makedirs(p["results_dir"], exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(p["results_dir"], f"ap_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(dict(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            max_segments=args.max_segments,
            rows=rows,
        ), f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
