#!/usr/bin/env bash
# CLEAR-MOT (MOTA / MOTP) benchmark on JRDB's held-out test sequences, using
# real per-sequence identity labels (no pseudo-linking, unlike
# mot_benchmark_drowv2). See config.yaml's header for prerequisites --
# nothing here downloads or extracts JRDB automatically.
#
#   1. jrdb_gt.py   flattens labeled frames + real label_id identities
#   2. replay.py    replays every scan through detector + tracker
#   3. score.py     associates against ground truth and prints MOTA / MOTP
#
# Usage:
#   ./run_mot_benchmark.sh                  # both trackers, both pipelines
#   ./run_mot_benchmark.sh --smoke          # 2 sequences, checks plumbing
#   ./run_mot_benchmark.sh --tracker norfair
#   ./run_mot_benchmark.sh --method kf-pipelined
#   ./run_mot_benchmark.sh -- --max-segments 3 --conf 0.5
#                                           # everything after -- goes to replay.py
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
MB="$REPO_ROOT/mot_benchmark_jrdb"

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

JRDB_TRAIN="$REPO_ROOT/dr_spaam/data/JRDB/train_dataset"
if [[ ! -d "$JRDB_TRAIN" ]]; then
  echo "missing $JRDB_TRAIN" >&2
  echo "JRDB must be downloaded (login required at https://jrdb.erc.monash.edu/)" >&2
  echo "and the combined laser scan extracted from rosbags -- see config.yaml's" >&2
  echo "PREREQUISITES header for the exact steps." >&2
  exit 1
fi

WEIGHTS="$REPO_ROOT/pretrained_weights/self_supervised_person_detection/ckpt_jrdb_ann_ft_dr_spaam_e20.pth"
[[ -f "$WEIGHTS" ]] || { echo "missing $WEIGHTS" >&2; exit 1; }

echo "=== 1/3  ground truth ==========================================================="
"$PY" "$MB/jrdb_gt.py"

echo
echo "=== 2/3  replay ==================================================================="
rm -f "$MB/out/tracks_jrdb_"*.npz
for t in $TRACKERS; do
  echo "--- tracker: $t ---"
  "$PY" "$MB/replay.py" --tracker "$t" --pipeline "$PIPELINES" "${EXTRA[@]}"
done

echo
echo "=== 3/3  score ===================================================================="
"$PY" "$MB/score.py"
