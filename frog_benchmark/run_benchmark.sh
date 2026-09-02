#!/usr/bin/env bash
# AP / precision-recall benchmark for DR-SPAAM(T=1) vs DR-SPAAM(T=5) on the
# FROG test split -- the mAP/mPeak-F1/mEER x {d=0.5m, d=0.3m} table format
# from FROG's own leaderboard. See frog_ap.py's docstring for the protocol.
#
# Requires mot_benchmark_frog/out/frog_16-41_gt.npz to already exist (run
# mot_benchmark_frog/frog_gt.py first if not).
#
# Usage:
#   ./run_benchmark.sh                        # both checkpoints, full test split
#   ./run_benchmark.sh --smoke                # 2 segments, checks plumbing
#   ./run_benchmark.sh --checkpoint "DR-SPAAM(T=1)"
#   ./run_benchmark.sh -- --max-segments 10    # everything after -- goes to frog_ap.py
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
FB="$REPO_ROOT/frog_benchmark"

EXTRA=()
CKPT=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)      EXTRA+=(--max-segments 2); shift ;;
    --checkpoint) CKPT=(--checkpoint "$2"); shift 2 ;;
    --)           shift; EXTRA+=("$@"); break ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

[[ -x "$PY" ]] || { echo "missing venv interpreter: $PY" >&2; exit 1; }
GT="$REPO_ROOT/mot_benchmark_frog/out/frog_16-41_gt.npz"
[[ -f "$GT" ]] || { echo "missing $GT -- run mot_benchmark_frog/frog_gt.py first" >&2; exit 1; }

cd "$FB"
"$PY" frog_ap.py "${CKPT[@]}" "${EXTRA[@]}"
