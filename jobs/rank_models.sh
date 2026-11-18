#!/bin/bash
# Rank finished checkpoints by multi-scale PQ on a fixed validation block.
ROOT=<PROJECT_ROOT>
PY=python
CARD=${1:-0}
shift || true

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$ROOT/src:${PYTHONPATH:-}
export TMPDIR=<TMP_DIR>
export OMP_NUM_THREADS=2
export ASCEND_RT_VISIBLE_DEVICES=$CARD
mkdir -p "$TMPDIR"
cd "$ROOT" || exit 1

RUNS=${*:-"u300 w36_200 ultra long32 nar20 pseudo_long res act_b24 act_w48 act_s640 act_s512"}

nohup $PY -u src/rank_runs.py --runs $RUNS --n 40 --scales 768,1024 \
  --out out/model_rank.json > logs/rank.log 2>&1 &
echo "launched ranking on card $CARD"
