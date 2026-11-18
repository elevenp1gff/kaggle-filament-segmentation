"""An honest ceiling: best achievable PQ without using ground truth at test time.

The 0.67 "oracle" number was inflated. It took `fgmask & gt` as the prediction,
which silently deletes every false-positive pixel (32.7% of the mask) by clipping
to GT. A real decoder cannot do that.

This computes a bound a real algorithm could approach:
  * start from the thresholded mask
  * assign pixels to instances with a GT-free rule (connected components here)
  * measure PQ

and then a second bound that is allowed one piece of GT information a real method
could plausibly estimate without labels: the *number* of instances per image (from
peak counting in the distance transform). Everything else is label-free.

The difference between these two tells whether closing the gap needs a better
splitting rule or a better probability field.
"""
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, "<PROJECT_ROOT>/src")
import data_fast as D
from cmp_single_pair import score_image

CACHE = "<CACHE_DIR>"


def pooled(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    isum = sum(r["iou_sum"] for r in rows)
    den = tp + 0.5 * fp + 0.5 * fn
    return (isum / den if den else 0.0), tp, fp, fn


def cc_decode(fg, thr, min_area):
    mask = (fg > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(lab == i).astype(np.uint8) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def peak_decode(fg, thr, min_area, k):
    """Split each component into k parts along the distance transform.

    k is the number of instances we believe are present. This is the one piece of
    information a label-free method could estimate by counting peaks.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    mask = (fg > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (lab == i).astype(np.uint8)
        if comp.sum() < 2000 or k <= 1:
            out.append(comp)
            continue
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        peaks = peak_local_max(dist, min_distance=25, num_peaks=k,
                              exclude_border=False, labels=comp)
        if len(peaks) <= 1:
            out.append(comp)
            continue
        markers = np.zeros(comp.shape, np.int32)
        for m, (yy, xx) in enumerate(peaks, 1):
            markers[yy, xx] = m
        ws = watershed(-dist, markers, mask=comp)
        for m in range(1, ws.max() + 1):
            part = (ws == m).astype(np.uint8)
            if part.sum() >= min_area:
                out.append(part)
    return out


def main():
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val = idx[24:84]

    gts, fgs, gt_counts = [], [], []
    for j, e in enumerate(val):
        pf = os.path.join(CACHE, "_prob", "ms_768_1024", "p%d.npy" % j)
        if not os.path.exists(pf):
            continue
        fgs.append(np.load(pf).astype(np.float32)[0])
        g = [D.polygons_to_window_mask(a["segmentation"], 0, 0, 2048, 2048)
             for a in e["anns"]]
        g = [m for m in g if m.sum() > 0]
        gts.append(g)
        gt_counts.append(len(g))
    print("%d images, %d GT" % (len(gts), sum(gt_counts)))

    print("\n=== label-free decoders (what we can actually ship) ===")
    for thr in (0.30, 0.10):
        for ma in (500, 200):
            rows = [score_image(gts[j], cc_decode(fgs[j], thr, ma))
                    for j in range(len(gts))]
            a = pooled(rows)
            print("  CC   thr=%.2f area=%3d: PQ %.4f (tp %d fp %d fn %d)"
                  % (thr, ma, a[0], a[1], a[2], a[3]))

    print("\n=== with the TRUE instance count (an upper bound on splitting) ===")
    for thr in (0.10,):
        rows = []
        for j in range(len(gts)):
            rows.append(score_image(
                gts[j], peak_decode(fgs[j], thr, 200, max(gt_counts[j], 1))))
        a = pooled(rows)
        print("  peak thr=%.2f k=true: PQ %.4f (tp %d fp %d fn %d)"
              % (thr, a[0], a[1], a[2], a[3]))


if __name__ == "__main__":
    main()
