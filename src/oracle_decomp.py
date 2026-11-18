"""Where exactly does the oracle advantage come from?

Split diagnostics showed grouping is NOT the problem: only 1% of components cover
two or more GT instances, and only 8% of GT instances are split. So the oracle's
superiority must come from pixel coverage, not instance assignment.

The oracle takes `fgmask & gt_instance` as its prediction. That means:
  predicted_area = |fgmask INTERSECT gt|
  intersection  = same quantity
so its IoU per instance = |fgmask & gt| / |fgmask | gt|.

This measures the per-instance IoU that this construction yields, and decomposes it:
how much of each GT is above threshold (recall ceiling), and how much above-threshold
pixel lies outside any GT (precision ceiling). If recall is high but precision is low,
the probabilities are smeared over background and a stricter threshold or a
shape prior is the fix. If both are high, the oracle number is inflated and the real
ceiling is lower.
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "<PROJECT_ROOT>/src")
import data_fast as D

CACHE = "<CACHE_DIR>"


def main():
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val = idx[24:84]

    for thr in (0.30, 0.10, 0.05):
        ious, recalls, outside_fracs, areas_gt, areas_pred = [], [], [], [], []
        for j, e in enumerate(val):
            pf = os.path.join(CACHE, "_prob", "ms_768_1024", "p%d.npy" % j)
            if not os.path.exists(pf):
                continue
            fg = np.load(pf).astype(np.float32)[0]
            mask = fg > thr
            gts = [D.polygons_to_window_mask(a["segmentation"], 0, 0, 2048, 2048)
                   for a in e["anns"]]
            gts = [m for m in gts if m.sum() > 0]
            if not gts:
                continue
            gt_all = np.zeros_like(mask, bool)
            for g in gts:
                gt_all |= g.astype(bool)
            outside = mask & ~gt_all
            outside_fracs.append(outside.sum() / max(mask.sum(), 1))
            for g in gts:
                gb = g.astype(bool)
                part = mask & gb
                if part.sum() == 0:
                    ious.append(0.0)
                    recalls.append(0.0)
                    continue
                union = part.sum() + gb.sum() - part.sum()   # part is subset of gt
                # prediction is `part`, so union = |part U gt| = |gt|
                ious.append(part.sum() / gb.sum())
                recalls.append(part.sum() / gb.sum())
                areas_gt.append(int(gb.sum()))
                areas_pred.append(int(part.sum()))

        ious = np.array(ious)
        out = np.array(outside_fracs)
        print("thr=%.2f  (%d instances)" % (thr, len(ious)))
        print("  GT pixels above threshold (recall ceiling): median %.2f, mean %.2f"
              % (np.median(recalls), np.mean(recalls)))
        print("  above-threshold pixels OUTSIDE any GT       : %.1f%% of the mask"
              % (out.mean() * 100))
        print("  per-instance IoU under oracle construction  : median %.3f mean %.3f"
              % (np.median(ious), np.mean(ious)))
        print("  instances with IoU > 0.5: %.1f%%" % ((ious > 0.5).mean() * 100))
        print()


if __name__ == "__main__":
    main()
