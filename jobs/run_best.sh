#!/bin/bash
# Generate the strongest submission: multi-scale x flip TTA.
# Usage: run_best.sh <card> <ckpt> <out.csv> <scales> <flips>
CARD=${1:-3}; CKPT=${2:-runs/u300/best.pt}; OUT=${3:-out/sub_best.csv}
SCALES=${4:-768,1024}; FLIPS=${5:-id,lr,ud}
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
echo "pinned $CKPT -> $PIN (sha $HASH)"

LOG=$ROOT/logs/best_${STAMP}.log
nohup $PY -u src/sub_best.py --ckpt "$PIN" --out "$OUT" --device npu:0 \
  --scales "$SCALES" --flips "$FLIPS" > "$LOG" 2>&1 &
echo "launched pid=$! log=$LOG"
echo "tail with: tail -f $LOG"
