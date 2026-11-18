"""Why does the decoder lose 0.37 of PQ that the probabilities already contain?

Oracle grouping reaches Q1 0.67 while the shipped decoder gets 0.30. The oracle is
allowed to use ground-truth instance identity to split pixels; the decoder must do it
from the probability field alone. So the entire gap is one question: how many
distinct instances are present, and where are their boundaries.

This measures the shape of that problem on real images:
  * how many connected components of the thresholded mask exist vs the GT count
  * how often two GT instances share one component (merge failure)
  * how often one GT instance is split across components (fragment failure)
Knowing which failure dominates tells us whether to add a splitting rule (watershed
on the distance transform, learned boundary, or peak counting) or a merging rule.
"""
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, "<PROJECT_ROOT>/src")
import data_fast as D

CACHE = "<CACHE_DIR>"


def analyse(gts, fg, thr):
    """Match GT instances to thresholded components and classify the failures."""
    mask = (fg > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [(lab == i).astype(bool) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 500]
    ng = len(gts)
    # how many components each GT instance touches
    touched = []
    for g in gts:
        gb = g.astype(bool)
        cnt = sum(1 for c in comps if np.count_nonzero(c & gb) > 0)
        touched.append(cnt)
    touched = np.array(touched) if touched else np.array([0])

    # how many GT instances each component covers
    per_comp = []
    for c in comps:
        k = sum(1 for g in gts if np.count_nonzero(c & g.astype(bool)) > 0)
        per_comp.append(k)
    per_comp = np.array(per_comp) if per_comp else np.array([0])

    return {
        "n_gt": ng,
        "n_comp": len(comps),
        "gt_missed": int((touched == 0).sum()),
        "gt_split": int((touched >= 2).sum()),
        "comp_merged": int((per_comp >= 2).sum()),
        "touched": touched,
        "per_comp": per_comp,
    }


def main():
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val = idx[24:84]

    for thr in (0.30, 0.10, 0.05):
        agg = {"n_gt": 0, "n_comp": 0, "gt_missed": 0, "gt_split": 0, "comp_merged": 0}
        for j, e in enumerate(val):
            pf = os.path.join(CACHE, "_prob", "ms_768_1024", "p%d.npy" % j)
            if not os.path.exists(pf):
                continue
            fg = np.load(pf).astype(np.float32)[0]
            g = [D.polygons_to_window_mask(a["segmentation"], 0, 0, 2048, 2048)
                 for a in e["anns"]]
            g = [m for m in g if m.sum() > 0]
            r = analyse(g, fg, thr)
            for k in agg:
                agg[k] += r[k]
        print("thr=%.2f" % thr)
        print("  GT instances      : %d" % agg["n_gt"])
        print("  components found  : %d" % agg["n_comp"])
        print("  GT missed entirely: %d (%.1f%%)"
              % (agg["gt_missed"], 100 * agg["gt_missed"] / max(agg["n_gt"], 1)))
        print("  GT split into 2+  : %d (%.1f%%)"
              % (agg["gt_split"], 100 * agg["gt_split"] / max(agg["n_gt"], 1)))
        print("  comp covers 2+ GT : %d (%.1f%% of comps) <- merge failure"
              % (agg["comp_merged"], 100 * agg["comp_merged"] / max(agg["n_comp"], 1)))
        print()


if __name__ == "__main__":
    main()
