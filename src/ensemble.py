"""Ensemble multiple trained models and decode instances once, at full resolution.

Rationale measured earlier: the ceiling for the current probabilities with ideal
grouping is PQ 0.478, while the decoder achieves 0.25. Averaging independent
models raises probability quality (the binding constraint) without changing the
decoder, and is cheap because inference is ~2 s/image.

Models can differ in crop size, so each predicts at its own resolution; the
resulting probability maps are resized to the 2048 canvas before averaging.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fast as D  # noqa: E402
from metrics import panoptic_quality  # noqa: E402


def normalize_u8(patch):
    x = patch.astype(np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return np.clip((x - med) / max(6.0 * mad, 1e-6), -1.0, 1.0)


def load_model(ckpt_path, device):
    from train import UNet2
    from train_res import ResUNet
    ck = torch.load(ckpt_path, map_location="cpu")
    a = ck.get("args", {})
    kind = "res" if "base" in a else "unet"
    if kind == "res":
        m = ResUNet(base=int(a.get("base", 32)))
    else:
        w = tuple(int(v) for v in str(a.get("width", "32,64,128,256,512")).split(","))
        m = UNet2(width=w)
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    return m, a, kind


@torch.no_grad()
def predict(model, arr_u8, size, stride, device, max_batch=16):
    img = normalize_u8(arr_u8)
    h, w = img.shape
    ph = max(size, int(math.ceil(h / stride) * stride))
    pw = max(size, int(math.ceil(w / stride) * stride))
    if (ph, pw) != (h, w):
        img = cv2.copyMakeBorder(img, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT)
    acc = np.zeros((2, ph, pw), np.float32)
    cnt = np.zeros((ph, pw), np.float32)
    origins = [(y, x) for y in range(0, ph - size + 1, stride)
               for x in range(0, pw - size + 1, stride)]
    dt = torch.device(device).type
    for i in range(0, len(origins), max_batch):
        ch = origins[i:i + max_batch]
        batch = np.stack([img[y:y + size, x:x + size] for y, x in ch])[:, None]
        t = torch.from_numpy(batch).to(device)
        with torch.autocast(dt, dtype=torch.bfloat16):
            out = model(t)
        pr = torch.sigmoid(out.float()).cpu().numpy()
        if pr.shape[-1] != size:
            pr = np.stack([cv2.resize(pr[k], (size, size)) for k in range(len(pr))])
        for (y, x), p in zip(ch, pr):
            acc[:, y:y + size, x:x + size] += p
            cnt[y:y + size, x:x + size] += 1
    acc /= np.maximum(cnt, 1e-6)[None]
    return acc[:, :h, :w]


def tta_predict(model, arr, size, stride, device, flips=(0, 1, 2, 3)):
    """Average predictions over identity plus 4 axis flips (physical symmetry)."""
    outs = []
    for k in flips:
        if k == 0:
            a, inv = arr, lambda x: x
        elif k == 1:
            a, inv = np.ascontiguousarray(arr[:, ::-1]), lambda x: x[:, ::-1]
        elif k == 2:
            a, inv = np.ascontiguousarray(arr[::-1]), lambda x: x[::-1]
        else:
            a, inv = np.ascontiguousarray(arr.T), lambda x: x.T
        p = predict(model, a, size, stride, device)
        outs.append(np.ascontiguousarray(inv(p)))
    return np.mean(outs, axis=0)


def decode(prob_fg, prob_bd, cfg):
    """Instance decode with the bd_support gate found useful earlier."""
    method = cfg.get("method", "semantic")
    thr = cfg.get("fg_thr", 0.5)
    min_area = cfg.get("min_area", 200)
    bd_sup = cfg.get("bd_support", 0.0)
    rim_k = cfg.get("rim_k", 5)

    fg = (prob_fg > thr).astype(np.uint8)
    if cfg.get("open_k"):
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (cfg["open_k"], cfg["open_k"])))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        m = (lab == i).astype(np.uint8)
        if bd_sup > 0:
            # ring just outside the candidate: real filaments have boundary response
            ring = cv2.dilate(m, np.ones((rim_k, rim_k), np.uint8)) & ~m
            if ring.sum() == 0:
                continue
            if float(prob_bd[ring.astype(bool)].mean()) < bd_sup:
                continue
        out.append(m)
    return out


def evaluate(models, entries, cache_index, device, decode_cfg, use_tta=False,
             verbose=True):
    gts_all, pr_all = [], []
    for e in entries:
        arr = np.load(cache_index[f"train/{e['file']}"])
        probs = []
        for m, a, kind in models:
            size = int(a.get("size", 512))
            stride = int(a.get("stride", size * 3 // 4))
            p = (tta_predict(m, arr, size, stride, device) if use_tta
                 else predict(m, arr, size, stride, device))
            probs.append(p)
        prob = np.mean(probs, axis=0)
        h, w = prob.shape[1:]
        g = []
        for ann in e["anns"]:
            mm = D.polygons_to_window_mask(ann["segmentation"], 0, 0, w, h)
            if mm.sum() > 0:
                g.append(mm)
        gts_all.append(g)
        pr_all.append(decode(prob[0], prob[1], decode_cfg))
    return panoptic_quality(gts_all, pr_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--cache-dir", default="<CACHE_DIR>")
    ap.add_argument("--val-n", type=int, default=24)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="out/ensemble_eval.json")
    args = ap.parse_args()

    from train_res import resolve_device
    device = resolve_device(args.device)
    cache_index = json.load(open(os.path.join(args.cache_dir, "index.json")))
    idx = D.attach_paths(D.build_index(), split="train")
    import random
    rng = random.Random(42)
    rng.shuffle(idx)
    val = idx[:args.val_n]

    models = []
    for p in args.ckpts:
        if not os.path.exists(p):
            print("missing", p)
            continue
        m, a, kind = load_model(p, device)
        models.append((m, a, kind))
        print("loaded %s kind=%s size=%s base=%s" % (
            os.path.basename(p), kind, a.get("size"), a.get("base")))
    if not models:
        print("no models loaded")
        return 1
    print(f"{len(models)} models, val {len(val)} images, tta={args.tta}")

    # small grid over the decode knobs that mattered before
    best = None
    for thr in (0.3, 0.4, 0.5):
        for min_area in (100, 200, 350):
            for bd_sup in (0.0, 0.2, 0.35, 0.5):
                cfg = dict(method="semantic", fg_thr=thr, min_area=min_area,
                           bd_support=bd_sup, rim_k=5)
                r = evaluate(models, val, cache_index, device, cfg, args.tta,
                             verbose=False)
                tag = f"thr={thr} area={min_area} bd={bd_sup}"
                print(f"  {tag:34s} PQ={r['pq']:.4f} SQ={r['sq']:.3f} "
                      f"RQ={r['rq']:.3f} tp={r['tp']} fp={r['fp']} fn={r['fn']}",
                      flush=True)
                if best is None or r["pq"] > best[0]["pq"]:
                    best = (r, cfg)
    print(f"\nBEST PQ={best[0]['pq']:.4f} with {best[1]}")
    json.dump({"best_pq": best[0]["pq"], "cfg": best[1],
               "ckpts": args.ckpts, "tta": args.tta},
              open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
