#!/bin/bash
# Launch pretrained-encoder runs. This is the one untried architectural change:
# every previous experiment kept a from-scratch encoder, and 20+ variants of the
# schedule/augmentation/decoder all landed on 0.32.
#
# ResNet34 is the primary (good balance of capacity and speed at 768 crops);
# ResNet18 is the fast control that confirms any gain comes from pretraining rather
# than from the different architecture.
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
  local card=$1 tag=$2 enc=$3; shift 3
  if pgrep -f "out runs/$tag" >/dev/null 2>&1; then echo "skip $tag"; return; fi
  ASCEND_RT_VISIBLE_DEVICES=$card nohup $PY -u src/train_smp.py --device npu:0 \
    --encoder "$enc" --weights imagenet --out "runs/$tag" "$@" \
    >> "logs/${tag}.log" 2>&1 &
  echo "launched $tag on card $card (encoder=$enc)"
}

# primary: ResNet34 pretrained, 200 epochs
launch 4 smp_r34 resnet34 --size 768 --epochs 200 --steps 120 --bs 4 --lr 5e-4 \
  --eval-every 50 --val-n 24

# control: ResNet18 pretrained, 200 epochs (faster, confirms the effect direction)
launch 6 smp_r18 resnet18 --size 768 --epochs 200 --steps 120 --bs 6 --lr 6e-4 \
  --eval-every 50 --val-n 24

sleep 90
for t in smp_r34 smp_r18; do
  echo "--- $t ---"
  grep -E "params|train |^ep |Error|error" "logs/$t.log" 2>/dev/null | tail -n 4
done
