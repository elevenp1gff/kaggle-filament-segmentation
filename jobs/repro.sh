#!/bin/bash
# Run the isolated repro and report a clear pass/fail verdict.
ROOT=<PROJECT_ROOT>
PY=python
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$ROOT/src:${PYTHONPATH:-}
export TMPDIR=<TMP_DIR>
export OMP_NUM_THREADS=2
export ASCEND_RT_VISIBLE_DEVICES=3
cd "$ROOT" || exit 1

OUT=$ROOT/logs/repro_out.txt
$PY - > "$OUT" 2>&1 <<'PYEOF'
import sys, json, random, traceback
sys.path.insert(0, "<PROJECT_ROOT>/src")
try:
    import torch, torch_npu
    import numpy as np
    import data_fast as D
    from train_res import PatchSource, build_targets

    cache_index = json.load(open("<CACHE_DIR>/index.json"))
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42); rng.shuffle(idx)
    train = idx[24:]
    for e in train:
        e["split"] = "train"

    src = PatchSource(train, 768, cache_index, pos_prob=0.8, aug=True, elastic=True)
    print("RESULT patchsource_ok paths=%d" % len(src.paths))

    for i in range(3):
        patch, insts = src.get("train")
    print("RESULT samples_ok patch=%s insts=%d dtypes=%s"
          % (patch.shape, len(insts), patch.dtype))

    y = build_targets(insts, 768)
    print("RESULT targets_ok shape=%s" % (y.shape,))

    # exercise the elastic path explicitly many times
    from train_res import augment
    for _ in range(20):
        p2, i2 = augment(patch.astype(np.uint8), insts, src.rng, elastic=True)
    print("RESULT elastic_ok 20x, out=%s" % (p2.shape,))

    from train_res import ResUNet
    m = ResUNet(base=32).npu()
    x = torch.from_numpy(patch[None, None].copy()).npu()
    with torch.autocast("npu", dtype=torch.bfloat16):
        out = m(x)
    print("RESULT forward_ok %s" % (tuple(out.shape),))
    print("RESULT ALL_OK")
except Exception:
    print("RESULT FAILED")
    traceback.print_exc()
PYEOF

grep -E "^RESULT|TypeError|ValueError|RuntimeError" "$OUT" | tail -n 12
