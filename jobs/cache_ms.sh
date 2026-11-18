#!/bin/bash
# Cache multi-scale (768+1024) averaged probabilities for the 100 validation
# images, so hard-quartile tuning can sweep configs without re-running inference.
ROOT=<PROJECT_ROOT>
PY=python
CARD=${1:-0}
CKPT=${2:-runs/u300/best.pt}
TAG=${3:-ms_768_1024}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$ROOT/src:${PYTHONPATH:-}
export TMPDIR=<TMP_DIR>
export OMP_NUM_THREADS=2
export ASCEND_RT_VISIBLE_DEVICES=$CARD
mkdir -p "$TMPDIR" "$ROOT/cache/blackhat_k41/_prob/$TAG"
cd "$ROOT" || exit 1

$PY - "$CKPT" "$TAG" > "logs/mscache_$TAG.log" 2>&1 <<'PYEOF'
import sys, os, json, random
sys.path.insert(0, "<PROJECT_ROOT>/src")
import numpy as np, torch
try:
    import torch_npu
except Exception:
    pass
import data_fast as D
from ensemble import load_model, predict
from train_res import resolve_device

ckpt, tag = sys.argv[1], sys.argv[2]
cache_dir = "<CACHE_DIR>"
out_dir = os.path.join(cache_dir, "_prob", tag)
os.makedirs(out_dir, exist_ok=True)

device = resolve_device("npu:0")
model, a, kind = load_model(ckpt, device)
print("model", kind, "from", ckpt, flush=True)

cache_index = json.load(open(os.path.join(cache_dir, "index.json")))
idx = D.attach_paths(D.build_index(), split="train")
rng = random.Random(42); rng.shuffle(idx)
val = idx[24:124]

scales = (768, 1024)
for j, e in enumerate(val):
    pf = os.path.join(out_dir, "p%d.npy" % j)
    if os.path.exists(pf):
        continue
    arr = np.load(cache_index["train/%s" % e["file"]])
    probs = [predict(model, arr, s, int(s * 0.75), device) for s in scales]
    np.save(pf, np.mean(probs, axis=0).astype(np.float16))
    if (j + 1) % 10 == 0:
        print("  %d/%d" % (j + 1, len(val)), flush=True)
print("DONE", tag)
PYEOF

tail -n 3 "logs/mscache_$TAG.log" | grep -vE "owner|warn"
