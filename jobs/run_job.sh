#!/bin/bash
# Launch one training/eval job on a single NPU card.
# Usage: run_job.sh <card> <log_tag> <script.py> [args...]
set -o pipefail

CARD=${1:?card required}
TAG=${2:?tag required}
SCRIPT=${3:?script required}
shift 3

ROOT=<PROJECT_ROOT>
PY=python
STAMP=$(date +%m%d_%H%M%S)
LOG=$ROOT/logs/${TAG}_${STAMP}.log

# source before any 'set -u': set_env.sh touches unset vars
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# thread discipline: 192 cores shared by 7 users
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=$ROOT/src:${PYTHONPATH:-}
export TMPDIR=<TMP_DIR>
ulimit -n 65535 2>/dev/null || true
mkdir -p "$TMPDIR"

# re-check the card right before committing to it
USED=$(npu-smi info 2>/dev/null | tr -d ' ' | awk -v c="$CARD" '
  $0 ~ "^\\|"c" " {f=1; next}
  f && /\/65536/ {split($0,a,"/"); gsub(/[^0-9]/,"",a[1]); print a[1]; exit}')
if [ -n "${USED:-}" ] && [ "${USED:-0}" -gt 20000 ] 2>/dev/null; then
  echo "card $CARD busy (HBM ${USED}MB), aborting" | tee "$LOG"
  exit 90
fi

cd "$ROOT"
export ASCEND_RT_VISIBLE_DEVICES=$CARD
export DEVICE=npu:0

echo "card=$CARD tag=$TAG script=$SCRIPT start=$(date '+%F %T')" | tee "$LOG"
$PY -u "$SCRIPT" "$@" 2>&1 | tee -a "$LOG"
echo "EXIT=$? end=$(date '+%F %T')" | tee -a "$LOG"
