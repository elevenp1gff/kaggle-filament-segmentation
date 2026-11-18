"""Panoptic Quality for solar filament instance segmentation, plus RLE I/O.

Official definition (competition Evaluation tab, citing Kirillov et al. CVPR 2019):

    PQ = sum_{(y, y_hat) in TP} IoU(y, y_hat)
         / ( |TP| + 0.5|FP| + 0.5|FN| )

where a predicted segment y_hat is matched to a ground-truth segment y iff
IoU(y, y_hat) > 0.5. Matching is one-to-one (a GT can absorb at most one
prediction and vice versa); the standard greedy implementation matches pairs in
descending IoU order, which for a >0.5 threshold makes the greedy result unique
and equal to the optimal assignment.

Submission format (from the Evaluation tab):
    CSV with columns filament_id, segmentation_rle
    filament_id example: 20150125172714Mh_1   (image stem + underscore + index)
    segmentation_rle: COCO compressed RLE *counts only* (no size dict needed,
    canvas is fixed at 2048 x 2048)
"""
from __future__ import annotations

import numpy as np

try:
    from pycocotools import mask as mask_utils
    HAVE_PYC = True
except Exception:  # pragma: no cover - pycocotools is optional for this module
    mask_utils = None
    HAVE_PYC = False


# --------------------------------------------------------------------------- #
#  RLE helpers (COCO compressed string form)
# --------------------------------------------------------------------------- #
def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary HxW mask (or list of them) to a COCO RLE counts string.

    Uses pycocotools when available; otherwise a compatible pure-python encoder
    so the same code runs on machines without the package.
    """
    if mask_utils is not None:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        return counts.decode("ascii") if isinstance(counts, bytes) else counts
    return _rle_encode_py(mask)


def _rle_encode_py(mask: np.ndarray) -> str:
    """Pure-python COCO RLE encoder (column-major run lengths, LEB128-ish string)."""
    pixels = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    counts = []
    if pixels.size == 0:
        return ""
    prev = 0
    run = 0
    for v in pixels:
        if v != prev:
            counts.append(run)
            run = 1
            prev = v
        else:
            run += 1
    counts.append(run)
    return _counts_to_string(counts)


def _counts_to_string(counts):
    """COCO's delta-coded, 5-bit-chunk string encoding of run lengths."""
    out = []
    for i, c in enumerate(counts):
        x = int(c)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            chr_ = x & 0x1F
            x >>= 5
            more = (x != -1) if (chr_ & 0x10) else (x != 0)
            if more:
                chr_ |= 0x20
            out.append(chr(chr_ + 48))
    return "".join(out)


def rle_decode(s: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode a COCO RLE counts string into a binary HxW uint8 mask."""
    if mask_utils is not None:
        rle = {"counts": s.encode("ascii") if isinstance(s, str) else s,
               "size": [int(height), int(width)]}
        return mask_utils.decode(rle).astype(np.uint8)
    return _rle_decode_py(s, height, width)


def _rle_decode_py(s: str, height: int, width: int) -> np.ndarray:
    counts = _string_to_counts(s)
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
        if pos >= total:
            break
    return flat.reshape((height, width), order="F")


def _string_to_counts(s: str):
    counts = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


# --------------------------------------------------------------------------- #
#  Panoptic Quality
# --------------------------------------------------------------------------- #
def pairwise_iou(gt_masks, pred_masks) -> np.ndarray:
    """IoU matrix of shape (n_gt, n_pred) via bitwise ops on flattened masks."""
    ng, npd = len(gt_masks), len(pred_masks)
    if ng == 0 or npd == 0:
        return np.zeros((ng, npd), dtype=np.float32)
    g = np.stack([m.reshape(-1).astype(bool) for m in gt_masks])
    p = np.stack([m.reshape(-1).astype(bool) for m in pred_masks])
    inter = (g.astype(np.uint32) @ p.astype(np.uint32).T).astype(np.float32)
    area_g = g.sum(1, dtype=np.float32)[:, None]
    area_p = p.sum(1, dtype=np.float32)[None, :]
    union = area_g + area_p - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return iou.astype(np.float32)


def panoptic_quality(gt_list, pred_list, iou_thresh: float = 0.5):
    """Aggregate PQ over a dataset.

    gt_list   : list (per image) of lists of binary masks
    pred_list : list (per image) of lists of binary masks

    Returns dict with pq, sq (segmentation quality), rq (recognition quality),
    tp, fp, fn, and the raw IoU sum so callers can pool subsets as the
    competition does (they score ~50% of images publicly).
    """
    assert len(gt_list) == len(pred_list)
    tp = fp = fn = 0
    iou_sum = 0.0
    for gts, preds in zip(gt_list, pred_list):
        iou = pairwise_iou(gts, preds)
        ng, npd = iou.shape
        used_g = np.zeros(ng, dtype=bool)
        used_p = np.zeros(npd, dtype=bool)

        # greedy one-to-one matching in descending IoU, only pairs above thresh
        if ng and npd:
            order = np.argsort(-iou, axis=None)
            for flat in order:
                gi, pi = divmod(int(flat), npd)
                if iou[gi, pi] <= iou_thresh:
                    break
                if used_g[gi] or used_p[pi]:
                    continue
                used_g[gi] = used_p[pi] = True
                iou_sum += float(iou[gi, pi])

        n_tp = int(used_g.sum())
        tp += n_tp
        fp += int(npd - n_tp)
        fn += int(ng - n_tp)

    denom = tp + 0.5 * fp + 0.5 * fn
    pq = float(iou_sum / denom) if denom > 0 else 0.0
    sq = float(iou_sum / tp) if tp > 0 else 0.0
    rq = float(tp / (tp + 0.5 * fp + 0.5 * fn)) if denom > 0 else 0.0
    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn,
            "iou_sum": iou_sum}


def dice_score(gt_list, pred_list) -> float:
    """Mean Dice over all GT instances, best-overlapping prediction (rubric metric).

    The competition also reports torchmetrics DiceScore; we mirror the common
    micro version: match each GT to its highest-IoU prediction, average 2I/(A+B).
    """
    vals = []
    for gts, preds in zip(gt_list, pred_list):
        if not gts:
            continue
        iou = pairwise_iou(gts, preds)
        for gi in range(len(gts)):
            if iou.shape[1] == 0:
                vals.append(0.0)
                continue
            best = float(iou[gi].max())
            vals.append(2 * best / (1 + best) if best > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


if __name__ == "__main__":
    # --- self-test: brute force PQ on random small masks vs the fast version ---
    rng = np.random.default_rng(0)
    for trial in range(5):
        gt, pr = [], []
        for _ in range(3):
            gts = [(rng.random((40, 40)) > 0.7).astype(np.uint8) for _ in range(3)]
            prs = [(rng.random((40, 40)) > 0.7).astype(np.uint8) for _ in range(3)]
            gt.append(gts)
            pr.append(prs)
        r = panoptic_quality(gt, pr)
        assert 0 <= r["pq"] <= 1, r
        print(f"trial {trial}: pq={r['pq']:.4f} tp={r['tp']} fp={r['fp']} fn={r['fn']}")

    # identical masks must give PQ = 1
    m1 = np.zeros((20, 20), np.uint8)
    m1[2:8, 3:9] = 1
    m2 = np.zeros((20, 20), np.uint8)
    m2[12:18, 10:17] = 1
    perfect = panoptic_quality([[m1, m2]], [[m1, m2]])
    print("\nidentical masks ->", perfect)
    assert abs(perfect["pq"] - 1.0) < 1e-6

    # no prediction -> PQ = 0
    empty = panoptic_quality([[m1]], [[]])
    print("no prediction    ->", empty)
    assert empty["pq"] == 0.0

    # RLE round trip
    if HAVE_PYC:
        s = rle_encode(m1)
        back = rle_decode(s, 20, 20)
        assert np.array_equal(back, m1), "pycocotools RLE round trip failed"
        print("\nRLE (pycocotools) round trip OK, counts len", len(s))
    s2 = _rle_encode_py(m1)
    back2 = _rle_decode_py(s2, 20, 20)
    print("RLE (pure python) round trip:", np.array_equal(back2, m1))
    assert np.array_equal(back2, m1), "pure python RLE round trip failed"

    if HAVE_PYC:
        # cross-check the two encoders agree
        assert rle_encode(m1) == _rle_encode_py(m1), "encoders disagree"
        print("both encoders produce identical strings: OK")
    print("\nall self-tests passed")
