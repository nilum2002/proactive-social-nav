#!/usr/bin/env bash
# CLEAR-MOT (MOTA / MOTP) benchmark on the FROG test split at a simulated
# 20 Hz scan rate (every 2nd scan of the native 40 Hz stream). See
# config.yaml's header for the decimation method and its caveats, and
# mot_benchmark_frog/run_mot_benchmark.sh for the 40 Hz baseline to compare
# against.
#
# Reuses mot_benchmark_frog/frog_gt.py's ground truth -- no separate GT step:
#   1. replay.py   replays the decimated scan stream through detector + tracker
#   2. score.py     associates against the same-decimated ground truth rows
#
# Usage:
#   ./run_mot_benchmark.sh                  # both trackers, both pipelines
#   ./run_mot_benchmark.sh --smoke          # 2 segments, checks plumbing
#   ./run_mot_benchmark.sh --tracker norfair
#   ./run_mot_benchmark.sh --method kf-pipelined
#   ./run_mot_benchmark.sh -- --max-segments 5 --conf 0.5
#                                           # everything after -- goes to replay.py
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
MB="$REPO_ROOT/mot_benchmark_frog_20Hz"

TRACKERS="kf norfair"
PIPELINES="both"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)     EXTRA+=(--max-segments 2 --conf 0.5); shift ;;
    --tracker)   TRACKERS="$2"; shift 2 ;;
    --pipeline)  PIPELINES="$2"; shift 2 ;;
    --method)
      m="$2"; shift 2
      case "$m" in
        kf|norfair)                           TRACKERS="$m" ;;
        kf-sequential|kf-pipelined)           TRACKERS="kf";      PIPELINES="${m#kf-}" ;;
        norfair-sequential|norfair-pipelined) TRACKERS="norfair"; PIPELINES="${m#norfair-}" ;;
        *) echo "unknown --method '$m' (want kf|norfair|kf-sequential|kf-pipelined|norfair-sequential|norfair-pipelined)" >&2
           exit 1 ;;
      esac ;;
    --)          shift; EXTRA+=("$@"); break ;;
    *)           EXTRA+=("$1"); shift ;;
  esac
done

[[ -x "$PY" ]] || { echo "missing venv interpreter: $PY" >&2; exit 1; }
GT="$REPO_ROOT/mot_benchmark_frog/out/frog_16-41_gt.npz"
[[ -f "$GT" ]] || { echo "missing $GT -- run mot_benchmark_frog/frog_gt.py first" >&2; exit 1; }

echo "=== 1/2  replay (20 Hz simulated) ==============================================="
rm -f "$MB/out/tracks_"*.npz
for t in $TRACKERS; do
  echo "--- tracker: $t ---"
  "$PY" "$MB/replay.py" --tracker "$t" --pipeline "$PIPELINES" "${EXTRA[@]}"
done

echo
echo "=== 2/2  score ================================================================"
"$PY" "$MB/score.py"
