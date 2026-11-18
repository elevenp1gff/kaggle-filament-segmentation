"""Which local metric should decide between checkpoints, given that Q1 misled us once?

Track record for candidate selection:
  * decode tuning: local Q1 tracked the board exactly (0.30 then 0.32)
  * model swap: union300 had Q1 0.3299 vs u300's 0.3065, yet scored 0.30 vs 0.32

So Q1 is a valid target when only the decoder changes, but it does not rank models.
That means model selection needs a different local signal. This script computes, for
the checkpoints we have, several candidate metrics on the same block and reports which
one would have ranked u300 above union300 - the only case with a known board verdict.

Metrics tested:
  A pooled PQ with the shipped decode
  B Q1 PQ
  C pooled PQ at a low threshold (recall-leaning, closer to raw probability quality)
  D mean per-image PQ (equal weight per image rather than per instance)
  E pooled PQ restricted to images with many instances (dense scenes)
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
CFG = dict(method="semantic", fg_thr=0.30, min_area=500, bd_support=0.45, rim_k=5)
CFG_LO = dict(method="semantic", fg_thr=0.10, min_area=300, bd_support=0.20, rim_k=5)


def pooled(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    isum = sum(r["iou_sum"] for r in rows)
    den = tp + 0.5 * fp + 0.5 * fn
    return (isum / den if den else 0.0)


def per_image(rows):
    vals = []
    for r in rows:
        den = r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]
        vals.append(r["iou_sum"] / den if den else 0.0)
    return float(np.mean(vals))


def main():
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val = idx[24:84]

    gts, gt_counts = [], []
    for e in val:
        g = [D.polygons_to_window_mask(a["segmentation"], 0, 0, 2048, 2048)
             for a in e["anns"]]
        g = [m for m in g if m.sum() > 0]
        gts.append(g)
        gt_counts.append(len(g))
    counts = np.array(gt_counts)
    dense = set(np.argsort(-counts)[:len(counts) // 3].tolist())

    models = {"u300": "single_u300", "union300": "single_union300"}
    print("%d images; known board verdict: u300 (0.32) > union300 (0.30)\n" % len(val))

    results = {}
    for name, tag in models.items():
        d = os.path.join(CACHE, "_prob", tag)
        probs = []
        ok = True
        for j in range(len(val)):
            pf = os.path.join(d, "p%d.npy" % j)
            if not os.path.exists(pf):
                ok = False
                break
            probs.append(np.load(pf).astype(np.float32))
        if not ok:
            print("skip %s (cache missing)" % name)
            continue

        rows = [score_image(gts[j], decode(probs[j][0], probs[j][1], CFG))
                for j in range(len(val))]
        rows_lo = [score_image(gts[j], decode(probs[j][0], probs[j][1], CFG_LO))
                   for j in range(len(val))]

        def pqi(r):
            den = r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]
            return r["iou_sum"] / den if den else 0.0

        order = np.argsort([pqi(r) for r in rows])
        q1 = order[:len(order) // 4]

        results[name] = {
            "A pooled": pooled(rows),
            "B Q1": pooled([rows[i] for i in q1]),
            "C low-thr pooled": pooled(rows_lo),
            "D mean per-image": per_image(rows),
            "E dense-scene pooled": pooled([rows[i] for i in sorted(dense)]),
        }

    print("%-22s %10s %10s %s" % ("metric", "u300", "union300", "correct?"))
    for m in ("A pooled", "B Q1", "C low-thr pooled", "D mean per-image",
              "E dense-scene pooled"):
        if "u300" not in results or "union300" not in results:
            continue
        a = results["u300"][m]
        b = results["union300"][m]
        mark = "YES" if a > b else "no"
        print("%-22s %10.4f %10.4f %s" % (m, a, b, mark))
    print("\nThe metric that ranks u300 above union300 is the one to trust for")
    print("model selection (Q1 failed on this exact case).")


if __name__ == "__main__":
    main()
