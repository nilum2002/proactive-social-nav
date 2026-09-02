"""AP / precision-recall for a dr_spaam checkpoint on the DROWv2 test split.

This is the *primary* benchmark in this directory, and the only one that
matches a metric DROW/DR-SPAAM literature actually reports -- MOTA/MOTP need
persistent ground-truth identity, which DROWv2's public release does not have
(see pseudo_gt.py for the secondary, approximate MOT path).

Reproduces the official evaluation protocol from
dr_spaam/dr_spaam/model/dr_spaam_fn.py::_model_eval_fn, not the streaming,
persistent-memory style mot_benchmark/replay.py uses for FROG:

    for each annotated frame, take the last `num_scans` (T) consecutive scans,
    reset the gate, replay those T scans through the model fresh
    (inference=False), and score only the prediction for the last one.

That is what DROWHandle/DROWDataset feed the model during training and
validation, so it is what a checkpoint's "T=5" training regime was tuned
against -- not the single-scan `inference=True` persistent-memory mode
Detector.__call__ uses for live deployment. Using the wrong protocol here
would silently understate a temporal checkpoint's own accuracy.

The checkpoints available (frog_dataset/*.pth) were trained on FROG, not
DROW, so this measures cross-dataset generalization -- not a reproduction of
DROW's own published leaderboard numbers.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np
import torch

from drow_common import load_config, DEFAULT_CONFIG, list_sequences, Sequence, get_laser_phi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dr_spaam"))

from dr_spaam.model.drow_net import DrowNet
from dr_spaam.model.dr_spaam import DrSpaam
import dr_spaam.utils.utils as u
import dr_spaam.utils.precision_recall as pru


def load_model(ckpt_file, model_name, gpu):
    """Mirrors dr_spaam.detector.Detector.__init__'s model construction --
    same hardcoded hyperparams, since those are cutout-size/gate constants,
    not laser geometry (DROW's 450-pt scan vs FROG's 720-pt scan makes no
    difference here)."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--weights", help="override detector.weight_file")
    ap.add_argument("--model", choices=["DR-SPAAM", "DROW3"], help="override detector.model")
    ap.add_argument("--num-scans", type=int, help="override detector.num_scans (T)")
    ap.add_argument("--max-sequences", type=int, default=None, help="limit sequences (smoke test)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = cfg["paths"]
    d = cfg["detector"]
    weight_file = args.weights or d["weight_file"]
    if not os.path.isabs(weight_file):
        weight_file = os.path.join(p["repo_root"], weight_file)
    model_name = args.model or d["model"]
    num_scans = args.num_scans or d["num_scans"]
    stride = d["scan_stride"]
    gpu = bool(d["gpu"])

    scan_phi = get_laser_phi()
    model = load_model(weight_file, model_name, gpu)
    use_dr_spaam = model_name == "DR-SPAAM"

    seqs = list_sequences(p["data_dir"], p["split"])
    if args.max_sequences:
        seqs = seqs[: args.max_sequences]
    if not seqs:
        sys.exit(f"no sequences found under {p['data_dir']}/{p['split']}")

    ap_dir = p["ap_dir"]
    shutil.rmtree(ap_dir, ignore_errors=True)
    os.makedirs(os.path.join(ap_dir, "detections"), exist_ok=True)
    os.makedirs(os.path.join(ap_dir, "groundtruth"), exist_ok=True)

    for seq_path in seqs:
        seq = Sequence(seq_path)
        os.makedirs(os.path.join(ap_dir, "detections", seq.name), exist_ok=True)
        os.makedirs(os.path.join(ap_dir, "groundtruth", seq.name), exist_ok=True)
        print(f"{seq.name}: {len(seq.ann_ns)} annotated frames / {len(seq.scans)} scans")

        for k, scan_idx in enumerate(seq.ann_scan_idx):
            scans = seq.window(int(scan_idx), num_scans, stride=1)
            dets_xy, dets_cls, _ = predict(
                model, use_dr_spaam, scans, scan_phi, cfg["cutout_kwargs"], stride, gpu
            )
            frame_id = f"{k:06d}"

            det_str = pru.drow_detection_to_kitti_string(dets_xy, dets_cls, None)
            with open(os.path.join(ap_dir, "detections", seq.name, f"{frame_id}.txt"), "w") as f:
                f.write(det_str)

            wp = seq.wp[k]
            if len(wp) > 0:
                gts_xy = np.stack(u.rphi_to_xy(wp[:, 0], wp[:, 1]), axis=1)
                # DROWDataset always marks anns_valid_mask all-True for genuine
                # DROW ground truth (only pseudo-labels use partial validity),
                # so occluded is always 0 here -- see drow_dataset.py:53.
                gts_occluded = np.zeros(len(gts_xy), dtype=int)
                gt_str = pru.drow_detection_to_kitti_string(gts_xy, None, gts_occluded)
            else:
                gt_str = ""
            with open(os.path.join(ap_dir, "groundtruth", seq.name, f"{frame_id}.txt"), "w") as f:
                f.write(gt_str)

    sequences, res_03, res_05 = pru.evaluate_drow(ap_dir, verbose=True)

    os.makedirs(p["results_dir"], exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(p["results_dir"], f"ap_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(dict(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            weight_file=os.path.relpath(weight_file, p["repo_root"]),
            model=model_name,
            num_scans=num_scans,
            split=p["split"],
            rows=[
                dict(sequence=s, ap_0_3=float(r3["ap"]), peak_f1_0_3=float(r3["peak_f1"]),
                     eer_0_3=float(r3["eer"]), ap_0_5=float(r5["ap"]),
                     peak_f1_0_5=float(r5["peak_f1"]), eer_0_5=float(r5["eer"]))
                for s, r3, r5 in zip(sequences, res_03, res_05)
            ],
        ), f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
