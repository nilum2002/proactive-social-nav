#!/usr/bin/env python3
import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

METHODS = [("grpc", "gRPC"), ("udp", "UDP"), ("wifi", "WiFi")]
MOTIONS = [("static_robot", "static"), ("moving_robot", "moving")]
MODES = ["seq", "pipelined"]

METRICS = [
    ("e2e_lat_ms", "t_lat_ms_mean"),
    ("det_ms", "t_det_ms_mean"),
    ("inference_ms", "inference_ms_mean"),
    ("gpu_percent", "gpu_util_percent"),
]


def find_log(method, motion, mode):
    d = os.path.join(ROOT, method, motion, mode)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if (f.startswith("inf_server_service_log")
                and f.endswith(".csv")
                and not f.endswith("_summary.csv")):
            return os.path.join(d, f)
    return None


def trim(path, target):
    df = pd.read_csv(path)
    reached = df.index[df["scans_total"] >= target]
    if len(reached) == 0:
        return df.reset_index(drop=True)
    return df.loc[: reached[0]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark.csv"))
    ap.add_argument("--round", type=int, default=2)
    args = ap.parse_args()

    rows = []
    for method, mlabel in METHODS:
        for motion, molabel in MOTIONS:
            for mode in MODES:
                path = find_log(method, motion, mode)
                if path is None:
                    continue
                df = trim(path, args.scans)
                row = {"run": f"{mlabel} {molabel} {mode}"}
                for out_name, col in METRICS:
                    s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
                    row[f"{out_name}_mean"] = s.mean() if s.size else float("nan")
                    row[f"{out_name}_sd"] = s.std() if s.size > 1 else float("nan")
                rows.append(row)

    cols = ["run"] + [f"{n}_{s}" for n, _ in METRICS for s in ("mean", "sd")]
    out = pd.DataFrame(rows)[cols].round(args.round)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
