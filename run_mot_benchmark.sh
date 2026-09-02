#!/usr/bin/env bash
# CLEAR-MOT (MOTA / MOTP) benchmark on the FROG dataset.
#
# Offline and standalone: no ROS, no gRPC, no robot. It drives DR-SPAAM and the
# tracker directly, so it is deterministic and re-runnable.
#
#   1. frog_gt.py  builds ground truth WITH identities, by recovering the `idp`
#                  tracklet ids from the raw labelling-tool CSVs (the published
#                  HDF5 drops them) and aligning to HDF5 row order. It aborts on
#                  a mismatch -- do not score against misaligned ground truth.
#   2. replay.py   replays every scan through detector + tracker and dumps tracks
#   3. score.py    associates against ground truth and prints MOTA / MOTP
#
# Usage:
#   ./run_mot_benchmark.sh                  # both trackers, both pipelines
#   ./run_mot_benchmark.sh --smoke          # 2 segments, ~4 min, checks plumbing
#   ./run_mot_benchmark.sh --tracker norfair
#   ./run_mot_benchmark.sh --method kf-pipelined
#                                           # shorthand for --tracker kf --pipeline pipelined
#                                           # also: kf, norfair, kf-sequential,
#                                           # norfair-sequential, norfair-pipelined
#   ./run_mot_benchmark.sh -- --max-segments 5 --conf 0.5
#                                           # everything after -- goes to replay.py
#
# NOT `set -u`: harmless here, but kept consistent with the other run scripts,
# where ROS's setup.bash trips nounset.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"         # has torch, h5py, stonesoup, norfair
MB="$REPO_ROOT/mot_benchmark_frog"

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

echo "=== 1/3  ground truth (must report ALIGNMENT OK) ==============================="
"$PY" "$MB/frog_gt.py"

echo
echo "=== 2/3  replay ==============================================================="
# Stale dumps from an earlier run would silently reappear in the score table,
# since score.py reads every tracks_*.npz it finds.
rm -f "$MB/out/tracks_"*.npz
for t in $TRACKERS; do
  echo "--- tracker: $t ---"
  "$PY" "$MB/replay.py" --tracker "$t" --pipeline "$PIPELINES" "${EXTRA[@]}"
done

echo
echo "=== 3/3  score ================================================================"
"$PY" "$MB/score.py"
