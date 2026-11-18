"""Fast MAGFiLO patch loader.

Written after the first probe died with no traceback: the original
``make_patch`` rasterised every polygon onto a full 2048x2048 canvas per
annotation, which is ~4 MB per mask and blows up memory once the OS page cache
and the NPU host buffer overlap.

Everything here rasterises *inside the patch window only*: a polygon is shifted
and clipped to the crop, so peak cost is one patch-sized array regardless of the
source image size. Masks are identical to full-canvas rasterisation because the
intersection with the window is all an IoU ever uses.
"""
from __future__ import annotations

import collections
import json
import os

import cv2
import numpy as np

DEFAULT_ROOT = "<DATA_ROOT>"


# --------------------------------------------------------------------------- #
#  index: one record per physical file
# --------------------------------------------------------------------------- #
def build_index(root=DEFAULT_ROOT, dedup="richest"):
    """Return [{file,h,w,anns}] with exactly one entry per file on disk.

    The COCO file has 1154 image records for 707 files; duplicates re-annotate
    the same filaments (measured: 57.7% of cross-record pairs have IoU>0.5), so
    keeping one record avoids double counting. ``dedup`` picks which one:
      'richest' - most annotations
      'union'   - merge annotations from all records, de-duplicated by geometry
    """
    ann_path = os.path.join(root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    d = json.load(open(ann_path))
    by_name = collections.defaultdict(list)
    for im in d["images"]:
        by_name[im["file_name"]].append(im)
    ann_by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    out = []
    for name, recs in by_name.items():
        h, w = recs[0]["height"], recs[0]["width"]
        if dedup == "union":
            seen, merged = set(), []
            for r in recs:
                for a in ann_by_img.get(r["id"], []):
                    key = (round(a["area"], 1),
                           tuple(round(v, 1) for v in a["bbox"]))
                    if key not in seen:
                        seen.add(key)
                        merged.append(a)
            anns = merged
        else:
            best = max(recs, key=lambda r: len(ann_by_img.get(r["id"], [])))
            anns = ann_by_img.get(best["id"], [])
        out.append({"file": name, "h": h, "w": w, "anns": anns})
    out.sort(key=lambda x: x["file"])
    return out


# --------------------------------------------------------------------------- #
#  image handling
# --------------------------------------------------------------------------- #
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def radial_normalize(img, n_bins=64, smooth=5):
    """Remove limb darkening / vignetting so filament contrast is uniform.

    Builds a radial brightness profile from the image itself and divides by it.
    Uses ``reduceat`` on sorted radii instead of ``np.add.at`` for speed.
    """
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    idx = np.clip((r / r.max() * n_bins).astype(np.int32), 0, n_bins - 1).ravel()
    order = np.argsort(idx, kind="stable")
    flat = f.ravel()[order]
    sidx = idx[order]
    bounds = np.searchsorted(sidx, np.arange(n_bins + 1))
    sums = np.add.reduceat(flat, bounds[:-1])
    counts = np.diff(bounds).astype(np.float32)
    prof = sums / np.maximum(counts, 1.0)
    k = np.ones(smooth, np.float32) / smooth
    prof = np.convolve(prof, k, mode="same")
    prof = np.maximum(prof, 1.0)
    return f / prof[idx.reshape(h, w)]


# --------------------------------------------------------------------------- #
#  polygon -> patch-local mask
# --------------------------------------------------------------------------- #
def polygons_to_window_mask(seg, x0, y0, pw, ph, dilate=0):
    """Rasterise COCO polygon(s) into a (ph,pw) window at offset (x0,y0).

    Exact for any IoU: clipping the polygon to the window gives the same
    intersection with anything else inside that window.
    """
    m = np.zeros((ph, pw), np.uint8)
    polys = seg if (isinstance(seg, list) and seg and isinstance(seg[0], list)) else [seg]
    off = np.array([x0, y0], np.float32)
    for poly in polys:
        pts = np.asarray(poly, np.float32).reshape(-1, 2) - off
        # cheap reject: entire polygon outside the window
        if pts[:, 0].max() < 0 or pts[:, 0].min() > pw or \
           pts[:, 1].max() < 0 or pts[:, 1].min() > ph:
            continue
        if pts.shape[0] >= 3:
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m


def extract_window(entry, x0, y0, size, img_cache=None):
    """Crop a size x size window and return (image, instance masks, meta)."""
    path = entry["_path"]
    if img_cache is not None and path in img_cache:
        img = img_cache[path]
    else:
        img = load_gray(path)
        if img_cache is not None:
            img_cache[path] = img

    h, w = img.shape
    x0 = int(np.clip(x0, 0, max(0, w - size)))
    y0 = int(np.clip(y0, 0, max(0, h - size)))
    patch = img[y0:y0 + size, x0:x0 + size]

    insts = []
    for a in entry["anns"]:
        bx, by, bw, bh = a["bbox"]
        if bx + bw <= x0 or bx >= x0 + size or by + bh <= y0 or by >= y0 + size:
            continue
        m = polygons_to_window_mask(a["segmentation"], x0, y0, size, size)
        if m.sum() > 0:
            insts.append(m)
    return patch, insts, (x0, y0)


def normalize_patch(patch, lo_pct=1.0, hi_pct=99.0):
    """Robust per-patch intensity scaling to roughly [-0.5, 3]."""
    lo, hi = np.percentile(patch, [lo_pct, hi_pct])
    return np.clip((patch - lo) / max(float(hi - lo), 1e-6), -0.5, 3.0).astype(np.float32)


def boundary_map(insts, size, thick=3):
    """Instance borders, used to supervise an edge/boundary head."""
    bd = np.zeros((size, size), np.uint8)
    k = np.ones((thick, thick), np.uint8)
    for m in insts:
        bd |= cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return bd


def foreground_map(insts, size):
    if not insts:
        return np.zeros((size, size), np.uint8)
    return np.stack(insts).max(0)


# --------------------------------------------------------------------------- #
#  sampling
# --------------------------------------------------------------------------- #
def choose_window(entry, rng, size, mode="train", pos_prob=0.75):
    """Pick a window origin. Train mode biases toward annotated filaments."""
    h, w = entry["h"], entry["w"]
    maxx, maxy = max(1, w - size), max(1, h - size)
    if mode == "train" and entry["anns"] and rng.random() < pos_prob:
        a = entry["anns"][rng.randrange(len(entry["anns"]))]
        bx, by, bw, bh = a["bbox"]
        cx = bx + bw / 2 + rng.uniform(-size / 2, size / 2)
        cy = by + bh / 2 + rng.uniform(-size / 2, size / 2)
        return int(np.clip(cx - size / 2, 0, maxx)), int(np.clip(cy - size / 2, 0, maxy))
    return rng.randrange(maxx + 1), rng.randrange(maxy + 1)


def attach_paths(entries, root=DEFAULT_ROOT, split="train"):
    d = os.path.join(root, split, f"{split}_images")
    for e in entries:
        e["_path"] = os.path.join(d, e["file"])
    return entries


if __name__ == "__main__":
    import random
    import time

    idx = attach_paths(build_index(), split="train")
    rng = random.Random(0)
    print(f"index: {len(idx)} files, "
          f"{sum(len(e['anns']) for e in idx)} annotations")

    t0 = time.time()
    n_pos = 0
    tot_inst = 0
    for i in range(30):
        e = idx[i]
        x0, y0 = choose_window(e, rng, 512, "train")
        p, insts, _ = extract_window(e, x0, y0, 512)
        tot_inst += len(insts)
        n_pos += int(bool(insts))
    dt = time.time() - t0
    print(f"30 windows in {dt:.2f}s ({dt/30*1000:.1f} ms each), "
          f"positive {n_pos}/30, instances {tot_inst}")

    # radial normalisation sanity: contrast between filaments and background
    e = idx[0]
    raw = load_gray(e["_path"])
    ren = radial_normalize(raw)
    h, w = raw.shape
    insts = []
    for a in e["anns"]:
        m = polygons_to_window_mask(a["segmentation"], 0, 0, w, h)
        if m.sum():
            insts.append(m)
    fg = np.zeros((h, w), bool)
    for m in insts:
        fg |= m.astype(bool)
    bg = ~fg
    print(f"raw    fg mean {raw[fg].mean():.1f}  bg mean {raw[bg].mean():.1f}  "
          f"gap {raw[fg].mean()-raw[bg].mean():.1f}")
    print(f"radial fg mean {ren[fg].mean():.3f} bg mean {ren[bg].mean():.3f}  "
          f"ratio {ren[fg].mean()/max(ren[bg].mean(),1e-6):.3f}")
