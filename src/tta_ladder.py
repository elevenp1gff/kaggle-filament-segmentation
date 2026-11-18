"""Final decode A/B: how many TTA passes are actually worth it?

The shipped submission uses 2 scales x 3 flips = 6 forward passes per image. That
costs ~12 minutes on 180 images. Earlier small-sample tests (16-24 images) suggested
3 flips beat 4 and that multi-scale+TTA composes, but those samples were small and
the board never moved between 0.30 and 0.32 tiers.

This measures, on 60 held-out images, the full ladder of pass counts against the
same probabilities, so the marginal value of each extra pass is visible. If 4 passes
match 6, the submission gets 33% faster with no loss; if 6 is clearly better, the
current recipe is justified.
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


def pooled(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    isum = sum(r["iou_sum"] for r in rows)
    den = tp + 0.5 * fp + 0.5 * fn
    return (isum / den if den else 0.0), tp, fp, fn


def prep(a, f):
    if f == "id":
        return a
    if f == "lr":
        return np.ascontiguousarray(a[:, ::-1])
    if f == "ud":
        return np.ascontiguousarray(a[::-1])
    return np.ascontiguousarray(a.T)


def un(p, f):
    if f == "id":
        return p
    if f == "lr":
        return np.ascontiguousarray(p[:, :, ::-1])
    if f == "ud":
        return np.ascontiguousarray(p[:, ::-1])
    return np.ascontiguousarray(p.transpose(0, 2, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/u300/best.pt")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="out/tta_ladder.json")
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

    combos = {
        "768 only":            (("768", "id"),),
        "768 x lr+ud":         (("768", "id"), ("768", "lr"), ("768", "ud")),
        "768+1024 x id":       (("768", "id"), ("1024", "id")),
        "768+1024 x lr+ud":    (("768", "id"), ("768", "lr"), ("768", "ud"),
                                ("1024", "id"), ("1024", "lr"), ("1024", "ud")),
        "768+1024 x all4":     (("768", "id"), ("768", "lr"), ("768", "ud"),
                                ("768", "tp"), ("1024", "id"), ("1024", "lr"),
                                ("1024", "ud"), ("1024", "tp")),
    }
    needed = sorted(set(c for combo in combos.values() for c in combo))
    print("%d unique passes to compute" % len(needed), flush=True)

    cfg = dict(method="semantic", fg_thr=0.30, min_area=500, bd_support=0.45, rim_k=5)

    # compute each (scale, flip) once per image, keep in memory one image at a time
    rows = {k: [] for k in combos}
    t0 = time.time()
    for j in range(len(val)):
        store = {}
        for key in needed:
            s, f = key
            try:
                p = predict(model, prep(arrs[j], f), int(s), int(int(s) * 0.75), device)
                store[key] = un(p, f)
            except Exception:
                store[key] = None
        for name, combo in combos.items():
            ps = [store[k] for k in combo if store.get(k) is not None]
            if not ps:
                continue
            prob = np.mean(ps, axis=0)
            rows[name].append(score_image(gts[j], decode(prob[0], prob[1], cfg)))
        if (j + 1) % 10 == 0:
            el = time.time() - t0
            print("  %d/%d  %.0fs" % (j + 1, len(val), el), flush=True)

    print("\n=== TTA ladder (Q1-style decode, %d images) ===" % len(val))
    base = None
    out = {}
    for name in combos:
        if not rows[name]:
            continue
        r = pooled(rows[name])
        out[name] = r[0]
        if base is None:
            base = r[0]
        print("  %-20s PQ=%.4f (%+.4f) tp=%d fp=%d fn=%d"
              % (name, r[0], r[0] - base, r[1], r[2], r[3]))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
