"""Compute per-image difficulty correctly, only for images we have cached probs for.

The failed attempt walked all 707 training images but the single-model cache holds
probability files only for the 100-image validation window (indices 0..99 of the
cropped split). Everything past that read a missing or wrong file and scored 0,
which made every weight degenerate.

This version:
  * rebuilds the same split as the cache build (rng seed 42, val = idx[24:124])
  * skips any image without a cache file instead of silently scoring 0
  * asserts that a plausible fraction of images were scored before writing weights
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "<PROJECT_ROOT>/src")
import data_fast as D
from cmp_single_pair import score_image
from ensemble import decode

CACHE = "<CACHE_DIR>"
CFG = dict(method="semantic", fg_thr=0.25, min_area=400, bd_support=0.45, rim_k=5)


def main():
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    # the cache was built over this exact window
    val = idx[24:124]

    prob_dir = os.path.join(CACHE, "_prob", "single_u300")
    difficulty = {}
    skipped = 0
    for j, e in enumerate(val):
        pf = os.path.join(prob_dir, "p%d.npy" % j)
        if not os.path.exists(pf):
            skipped += 1
            continue
        prob = np.load(pf).astype(np.float32)
        g = [D.polygons_to_window_mask(a["segmentation"], 0, 0, 2048, 2048)
             for a in e["anns"]]
        g = [m for m in g if m.sum() > 0]
        r = score_image(g, decode(prob[0], prob[1], CFG))
        den = r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]
        difficulty[e["file"]] = r["iou_sum"] / den if den else 0.0

    vals = np.array(list(difficulty.values()))
    print("scored %d images (%d skipped), difficulty min %.3f median %.3f max %.3f"
          % (len(difficulty), skipped, vals.min(), np.median(vals), vals.max()))
    assert len(difficulty) >= 50, "too few images scored; cache keys must be wrong"
    assert vals.std() > 1e-6, "all difficulties identical; nothing to weight"

    # weight = 1/(pq+eps) clipped at 4x the median, then normalised
    w = 1.0 / (vals + 0.05)
    w = np.clip(w, 0, np.median(w) * 4.0)
    w = w / w.sum()
    weights = {n: float(x) for n, x in zip(difficulty.keys(), w)}

    thr = np.percentile(vals, 25)
    hard = [n for n in difficulty if difficulty[n] <= thr]
    print("hardest quartile: %d images, %.1f%% of draws (uniform would be 25%%)"
          % (len(hard), sum(weights[n] for n in hard) * 100))
    print("weight range: %.5f .. %.5f (ratio %.1fx)"
          % (w.min(), w.max(), w.max() / max(w.min(), 1e-12)))

    out = os.path.join(CACHE, "hard_weights.json")
    json.dump(weights, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
