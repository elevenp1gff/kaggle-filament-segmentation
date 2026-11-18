"""Decide between two-scale and three-scale with a paired test on more images.

A 30-image probe showed 768+1024+1280 (0.4349) edging 768+1024 (0.4334), but an
earlier 60-image test showed the opposite (0.4417 vs 0.4439). The two results
disagree, so the difference is inside noise at these sample sizes. Adding a scale
costs 50% more inference, so it should only ship if the effect is real.

This runs the comparison image-by-image on more images and reports the paired win
count, which is far less sensitive to a few extreme images than the pooled ratio.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

sys.path.insert(0, "<PROJECT_ROOT>/src")
import data_fast as D
from cmp_single_pair import score_image
from ensemble import decode, load_model, predict

CACHE = "<CACHE_DIR>"
CFG = dict(method="semantic", fg_thr=0.30, min_area=500, bd_support=0.45, rim_k=5)


def pooled(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    isum = sum(r["iou_sum"] for r in rows)
    den = tp + 0.5 * fp + 0.5 * fn
    return (isum / den if den else 0.0)


def pqi(r):
    den = r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]
    return r["iou_sum"] / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/u300/best.pt")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    from train_res import resolve_device
    device = resolve_device(args.device)
    cache_index = json.load(open(os.path.join(CACHE, "index.json")))
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val = idx[24:24 + args.n]

    model, a, kind = load_model(args.ckpt, device)
    print("model %s, %d images" % (kind, len(val)), flush=True)

    gts, arrs = [], []
    for e in val:
        g = [D.polygons_to_window_mask(x["segmentation"], 0, 0, 2048, 2048)
             for x in e["anns"]]
        gts.append([m for m in g if m.sum() > 0])
        arrs.append(np.load(cache_index["train/%s" % e["file"]]))

    rows2, rows3 = [], []
    t0 = time.time()
    for j in range(len(val)):
        try:
            p768 = predict(model, arrs[j], 768, 576, device)
            p1024 = predict(model, arrs[j], 1024, 768, device)
            p1280 = predict(model, arrs[j], 1280, 960, device)
        except Exception:
            continue
        a2 = (p768 + p1024) / 2
        a3 = (p768 + p1024 + p1280) / 3
        rows2.append(score_image(gts[j], decode(a2[0], a2[1], CFG)))
        rows3.append(score_image(gts[j], decode(a3[0], a3[1], CFG)))
        if (j + 1) % 10 == 0:
            print("  %d/%d  %.0fs" % (j + 1, len(val), time.time() - t0), flush=True)

    n = len(rows2)
    p2, p3 = pooled(rows2), pooled(rows3)
    wins = sum(1 for i in range(n) if pqi(rows3[i]) > pqi(rows2[i]))
    print("\n=== two-scale vs three-scale (%d images) ===" % n)
    print("  768+1024        PQ=%.4f" % p2)
    print("  768+1024+1280   PQ=%.4f (%+.4f)" % (p3, p3 - p2))
    print("  three-scale wins on %d/%d images (%.0f%%)" % (wins, n, 100 * wins / max(n, 1)))

    # bootstrap the difference
    rng2 = np.random.default_rng(0)
    d = []
    for _ in range(600):
        s = rng2.integers(0, n, n)
        d.append(pooled([rows3[i] for i in s]) - pooled([rows2[i] for i in s]))
    d = np.array(d)
    print("  paired bootstrap: mean %+.4f [p5 %+.4f, p95 %+.4f], P(3 better) %.0f%%"
          % (d.mean(), np.percentile(d, 5), np.percentile(d, 95), (d > 0).mean() * 100))


if __name__ == "__main__":
    main()
