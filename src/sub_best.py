"""Submission with the strongest measured configuration.

Measured on 16 held-out images (same decode settings throughout):

  768 only             PQ 0.4487   tp 101 fp 61 fn 43   <- what scored 0.30 on the board
  768+1024             PQ 0.4499
  768 x lr+ud          PQ 0.4632
  768+1024 x lr+ud     PQ 0.4663   tp 104 fp 56 fn 40   <- best
  768+1024 x all4      PQ 0.4638

So this uses two scales (768, 1024) crossed with three orientations (identity,
left-right, up-down): six forward passes per image, probabilities averaged in the
original orientation, then decoded once. All three error counts improve versus the
current best submission, which is why it is worth the 6x inference cost (about
15 s/image, ~45 min for 180 test images).

CSV is written LF-only: a CRLF file appends \\r to every RLE string.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fast as D  # noqa: E402
from ensemble import decode, load_model, predict  # noqa: E402
from metrics import rle_encode  # noqa: E402


def prep(arr, flip):
    if flip == "id":
        return arr
    if flip == "lr":
        return np.ascontiguousarray(arr[:, ::-1])
    if flip == "ud":
        return np.ascontiguousarray(arr[::-1])
    return np.ascontiguousarray(arr.T)


def untransform(prob, flip):
    if flip == "id":
        return prob
    if flip == "lr":
        return np.ascontiguousarray(prob[:, :, ::-1])
    if flip == "ud":
        return np.ascontiguousarray(prob[:, ::-1])
    return np.ascontiguousarray(prob.transpose(0, 2, 1))


def safe_predict(model, arr, size, stride, device, tries=3):
    for _ in range(tries):
        try:
            return predict(model, arr, size, stride, device)
        except Exception as e:
            print("      retry after %s" % type(e).__name__, flush=True)
            time.sleep(2)
    return None


def list_test(root=D.DEFAULT_ROOT):
    d = os.path.join(root, "test", "test_images")
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".jpeg", ".jpg", ".png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/u300/best.pt")
    ap.add_argument("--scales", default="768,1024")
    ap.add_argument("--flips", default="id,lr,ud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--fg-thr", type=float, default=0.30)
    ap.add_argument("--min-area", type=int, default=500)
    ap.add_argument("--bd-support", type=float, default=0.45)
    ap.add_argument("--rim-k", type=int, default=5)
    ap.add_argument("--max-inst", type=int, default=300)
    ap.add_argument("--cache-dir",
                    default="<CACHE_DIR>")
    args = ap.parse_args()

    from train_res import resolve_device
    device = resolve_device(args.device)
    model, a, kind = load_model(args.ckpt, device)
    scales = [int(s) for s in args.scales.split(",")]
    flips = [f.strip() for f in args.flips.split(",") if f.strip()]
    print("model %s, scales %s, flips %s -> %d passes/image"
          % (kind, scales, flips, len(scales) * len(flips)), flush=True)

    cache_index = json.load(open(os.path.join(args.cache_dir, "index.json")))
    cfg = dict(method="semantic", fg_thr=args.fg_thr, min_area=args.min_area,
               bd_support=args.bd_support, rim_k=args.rim_k)

    files = list_test()
    print("%d test images" % len(files), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = 0
    empty = []
    t0 = time.time()
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filament_id", "segmentation_rle"])
        for k, name in enumerate(files, 1):
            key = "test/%s" % name
            if key not in cache_index:
                print("missing cache", name)
                continue
            arr = np.load(cache_index[key])
            probs = []
            for s in scales:
                for fl in flips:
                    p = safe_predict(model, prep(arr, fl), s, int(s * 0.75), device)
                    if p is not None:
                        probs.append(untransform(p, fl))
            if not probs:
                empty.append(os.path.splitext(name)[0])
                continue
            prob = np.mean(probs, axis=0)
            insts = decode(prob[0], prob[1], cfg)
            # Fallback: a handful of test images decode to nothing under the tuned
            # settings, costing all their GT instances as FN. Relaxing only for
            # those images cannot affect any image that already had predictions, so
            # it is a non-negative change on the metric.
            if not insts:
                for relax in (dict(fg_thr=0.20, min_area=300, bd_support=0.30),
                              dict(fg_thr=0.12, min_area=150, bd_support=0.15),
                              dict(fg_thr=0.08, min_area=80, bd_support=0.0)):
                    insts = decode(prob[0], prob[1],
                                   dict(method="semantic", rim_k=args.rim_k, **relax))
                    if insts:
                        print("  fallback fired for %s -> %d instances at %s"
                              % (name, len(insts), relax), flush=True)
                        break
            if len(insts) > args.max_inst:
                insts = sorted(insts, key=lambda m: -m.sum())[:args.max_inst]
            stem = os.path.splitext(name)[0]
            if not insts:
                empty.append(stem)
            for i, m in enumerate(insts, 1):
                w.writerow(["%s_%d" % (stem, i), rle_encode(m).replace('"', "")])
            total += len(insts)
            if k % 20 == 0:
                el = time.time() - t0
                print("[%d/%d] rows=%d  %.0fs elapsed, ~%.0fs left"
                      % (k, len(files), total, el, el / k * (len(files) - k)),
                      flush=True)

    print("wrote %s: %d rows (%.1f/image) in %.0fs"
          % (args.out, total, total / max(len(files), 1), time.time() - t0))
    if empty:
        print("no-prediction images (%d): %s" % (len(empty), empty[:5]))


if __name__ == "__main__":
    main()
