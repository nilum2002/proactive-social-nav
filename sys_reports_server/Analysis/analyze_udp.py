#!/usr/bin/env python3
import argparse

from analysis_common import SCAN_TARGET, analyze_method


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scans", type=int, default=SCAN_TARGET,
                    help="number of scans to analyse from the start of each run")
    args = ap.parse_args()
    analyze_method("udp", target=args.scans)


if __name__ == "__main__":
    main()
