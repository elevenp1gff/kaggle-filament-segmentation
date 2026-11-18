#!/bin/bash
# Generate a multi-scale submission on one card.
# Usage: run_sub_ms.sh <card> <ckpt> <out.csv> <scales>
CARD=${1:?}; CKPT=${2:?}; OUT=${3:?}; SCALES=${4:-768,1024}
ROOT=<PROJECT_ROOT>
PY=python

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$ROOT/src:${PYTHONPATH:-}
export TMPDIR=<TMP_DIR>
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ASCEND_RT_VISIBLE_DEVICES=$CARD
mkdir -p "$TMPDIR"
cd "$ROOT" || exit 1

HASH=$(sha256sum "$CKPT" | cut -c1-12)
STAMP=$(date +%m%d_%H%M%S)
mkdir -p out/ckpt
PIN="out/ckpt/best_${STAMP}_${HASH}.pt"
cp -p "$CKPT" "$PIN"
echo "pinned $CKPT -> $PIN"

LOG=$ROOT/logs/subms_${STAMP}.log
$PY -u src/sub_ms.py --ckpt "$PIN" --out "$OUT" --device npu:0 \
  --scales "$SCALES" >> "$LOG" 2>&1
RC=$?
echo "EXIT=$RC"
grep -E "model |decode|test images|wrote|no-prediction" "$LOG" | tail -n 5
if [ $RC -eq 0 ]; then
  echo "rows: $(wc -l < "$OUT")"
  echo "$PIN $OUT scales=$SCALES $(sha256sum "$OUT" | cut -d' ' -f1)" >> out/submissions.log
fi
