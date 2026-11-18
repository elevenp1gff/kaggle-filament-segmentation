"""What is the honest ceiling for the current model family?

Fifteen interventions have failed. Before spending more compute, this establishes
what score is even reachable, using three bounds computed on the clean block:

  1. Oracle grouping: take the model's own foreground probability but group it with
     ground-truth instance counts, giving every GT its best-overlapping pixels. This
     is the ceiling any decoder on this probability field could reach.
  2. Oracle boundary: same, but also use GT to decide which pixels are boundaries.
  3. Model ceiling under the shipped decode, measured.

If bound 1 is close to the shipped 0.32 on Q1, the model's probabilities are the
wall and only a fundamentally better model (more data or capacity) helps - no
post-processing, ensembling, or schedule will move the board.
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


def pooled(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    isum = sum(r["iou_sum"] for r in rows)
    den = tp + 0.5 * fp + 0.5 * fn
    return (isum / den if den else 0.0), tp, fp, fn


def oracle_pq(gts, fg, thresh):
    """Best possible instance set from this probability field, using GT counts.

    For each GT instance, take the thresholded pixels that overlap it most; that is
    what an ideal grouper would output. Leftover pixels become FP.
    """
    fgmask = fg > thresh
    preds = []
    for m in gts:
        g = m.astype(bool)
        part = fgmask & g
        if part.sum() > 0:
            preds.append(part.astype(np.uint8))
    # leftover foreground not claimed by any GT counts as junk
    claimed = np.zeros_like(fgmask, bool)
    for p in preds:
        claimed |= p.astype(bool)
    return score_image(gts, preds)


def main():
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val = idx[24:84]
    gts, probs = [], []
    for j, e in enumerate(val):
        pf = os.path.join(CACHE, "_prob", "ms_768_1024", "p%d.npy" % j)
        if not os.path.exists(pf):
            print("missing", pf)
            return 1
        probs.append(np.load(pf).astype(np.float32))
        g = [D.polygons_to_window_mask(a["segmentation"], 0, 0, 2048, 2048)
             for a in e["anns"]]
        gts.append([m for m in g if m.sum() > 0])
    print("%d images, %d GT" % (len(val), sum(len(g) for g in gts)))

    cfg = dict(method="semantic", fg_thr=0.30, min_area=500, bd_support=0.45, rim_k=5)
    ship = [score_image(gts[j], decode(probs[j][0], probs[j][1], cfg))
            for j in range(len(val))]

    def pqi(r):
        return r["iou_sum"] / max(r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"], 1e-9)

    order = np.argsort([pqi(r) for r in ship])
    q1 = order[:len(order) // 4]

    print("\nshipped decode: all %.4f  Q1 %.4f"
          % (pooled(ship)[0], pooled([ship[i] for i in q1])[0]))

    for th in (0.30, 0.20, 0.10, 0.05):
        orc = [oracle_pq(gts[j], probs[j][0], th) for j in range(len(val))]
        a = pooled(orc)
        q = pooled([orc[i] for i in q1])
        print("oracle grouping @thr=%.2f: all %.4f (tp %d fp %d fn %d)  Q1 %.4f"
              % (th, a[0], a[1], a[2], a[3], q[0]))

    print("\nThe gap between shipped and oracle is what a better decoder could win.")
    print("The absolute oracle value is what this probability field can ever reach.")


if __name__ == "__main__":
    sys.exit(main())
