#!/bin/bash
# Launch coverage-aware training runs.
#
# Rationale (measured): random 768-px windows touch only 18.9% of GT instances in
# 400 draws, so ~81% of filaments are rarely or never seen during training while
# evaluation scores all of them. A grid at 1024/stride 768 reaches 99.7% coverage
# with 9 windows per image, and 768/stride 576 reaches 100% with 16.
#
# Two runs: 1024 crops (best coverage per window) and 768 crops (matches the
# proven u300 recipe, so the comparison isolates the sampler).
ROOT=<PROJECT_ROOT>
PY=python

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$ROOT/src:${PYTHONPATH:-}
export TMPDIR=<TMP_DIR>
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p "$TMPDIR" "$ROOT/logs"
cd "$ROOT" || exit 1

launch () {
  local card=$1 tag=$2; shift 2
  if pgrep -f "out runs/$tag" >/dev/null 2>&1; then echo "skip $tag"; return; fi
  ASCEND_RT_VISIBLE_DEVICES=$card nohup $PY -u src/train_cover.py --device npu:0 "$@" \
    --out "runs/$tag" >> "logs/${tag}.log" 2>&1 &
  echo "launched $tag on card $card"
}

# 1024 crop: 99.7% coverage with 9 windows/img, bs 2 to fit memory
launch 0 cov1024 --size 1024 --stride 768 --epochs 150 --steps 120 --bs 2 \
  --base 32 --lr 8e-4 --cover-ratio 0.5 --eval-every 30 --val-n 24

# 768 crop: same geometry as the proven u300 recipe, only the sampler changes
launch 3 cov768 --size 768 --stride 576 --epochs 200 --steps 120 --bs 4 \
  --base 32 --lr 8e-4 --cover-ratio 0.5 --eval-every 40 --val-n 24

sleep 25
pgrep -af "train_cover.py" | sed 's/.*--out //' | cut -d' ' -f1 | sort | uniq -c
