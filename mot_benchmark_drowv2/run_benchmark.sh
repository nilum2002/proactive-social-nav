#!/usr/bin/env bash
# DROWv2 test-split benchmark for the FROG-trained DR-SPAAM/DROW3 checkpoints.
#
# Two independent benchmarks, both offline and standalone (no ROS, no gRPC):
#
#   PRIMARY   drow_ap.py            AP / precision-recall, the actual published
#                                    DROW/DR-SPAAM metric. Windowed (T-scan,
#                                    gate-reset-per-sample) protocol, matching
#                                    dr_spaam/dr_spaam/model/dr_spaam_fn.py.
#
#   SECONDARY pseudo_gt.py           Approximate MOTA/MOTP. DROWv2 has no
#             + replay.py            ground-truth person identity, so
#             + score.py             pseudo_gt.py fakes tracklets by linking
#                                    nearest GT points across consecutive
#                                    annotated frames -- read its docstring
#                                    before trusting IDSW from this path.
#
# These checkpoints (frog_dataset/*.pth) were trained on FROG, not DROW, so
# both benchmarks measure cross-dataset generalization, not a reproduction of
# DROW's own published leaderboard.
#
# Usage:
#   ./run_benchmark.sh                 # both benchmarks, full test split
#   ./run_benchmark.sh --smoke         # 1 sequence, checks plumbing
#   ./run_benchmark.sh --ap-only
#   ./run_benchmark.sh --mot-only
#   ./run_benchmark.sh --tracker norfair --mot-only
#   ./run_benchmark.sh --method kf-pipelined --mot-only
#   ./run_benchmark.sh --mot-only -- --max-sequences 2
#                                       # everything after -- goes to replay.py
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
DW="$REPO_ROOT/mot_benchmark_drowv2"

RUN_AP=1
RUN_MOT=1
TRACKERS="kf"
PIPELINES="both"
SMOKE=()
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)     SMOKE=(--max-sequences 1); EXTRA+=(--max-sequences 1); shift ;;
    --ap-only)   RUN_MOT=0; shift ;;
    --mot-only)  RUN_AP=0; shift ;;
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
cd "$DW"

if [[ "$RUN_AP" == 1 ]]; then
  echo "=== AP / precision-recall (primary) ============================================"
  "$PY" drow_ap.py "${SMOKE[@]}"
  echo
fi

if [[ "$RUN_MOT" == 1 ]]; then
  echo "=== pseudo-MOT (secondary, approximate -- see pseudo_gt.py) ===================="
  echo "--- 1/3  pseudo ground truth ---"
  "$PY" pseudo_gt.py

  echo
  echo "--- 2/3  replay ---"
  rm -f out/drowv2_test_tracks_*.npz
  for t in $TRACKERS; do
    echo "tracker: $t"
    "$PY" replay.py --tracker "$t" --pipeline "$PIPELINES" "${EXTRA[@]}"
  done

  echo
  echo "--- 3/3  score ---"
  "$PY" score.py
fi
