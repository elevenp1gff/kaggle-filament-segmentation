"""Precompute blackhat-preprocessed images once so training never pays 527 ms.

Profiling: a full 2048x2048 blackhat costs 527 ms, so a bs=8 step spent ~4.2 s
in the loader while the NPU idled at 0%. Caching all 707 images as uint8 takes
~110 s and 3.0 GB, after which a window is a memcpy.

The cache is written as one .npy per image plus an index, because a single 3 GB
file read over NFS is slower to page in than individual files.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fast as D


def preprocess_array(gray, mode, ksize):
    if mode == "blackhat":
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, ker)
    if mode == "raw":
        return gray.astype(np.uint8)
    if mode == "local_z":
        f = gray.astype(np.float32)
        mu = cv2.GaussianBlur(f, (0, 0), 15)
        mu2 = cv2.GaussianBlur(f * f, (0, 0), 15)
        z = (f - mu) / np.sqrt(np.maximum(mu2 - mu * mu, 1.0))
        return np.clip(z * 64 + 128, 0, 255).astype(np.uint8)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="blackhat")
    ap.add_argument("--ksize", type=int, default=41)
    ap.add_argument("--out", default="<CACHE_ROOT>")
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()

    root = D.DEFAULT_ROOT
    todo = []
    for split in args.splits.split(","):
        d = os.path.join(root, split, f"{split}_images")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpeg", ".jpg", ".png")):
                todo.append((split, f, os.path.join(d, f)))

    outdir = os.path.join(args.out, f"{args.mode}_k{args.ksize}")
    os.makedirs(outdir, exist_ok=True)
    print(f"{len(todo)} images -> {outdir}")

    t0 = time.time()
    index = {}
    for i, (split, name, path) in enumerate(todo, 1):
        dst = os.path.join(outdir, f"{split}__{name}.npy")
        index_key = f"{split}/{name}"
        if os.path.exists(dst):
            index[index_key] = dst
            continue
        g = D.load_gray(path)
        arr = preprocess_array(g, args.mode, args.ksize)
        np.save(dst, arr)
        index[index_key] = dst
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el:.0f}s  eta {el/i*(len(todo)-i):.0f}s",
                  flush=True)

    with open(os.path.join(outdir, "index.json"), "w") as f:
        json.dump(index, f)
    total = sum(os.path.getsize(p) for p in index.values())
    print(f"done in {time.time()-t0:.0f}s, {total/1e9:.2f} GB, "
          f"{len(index)} files")


if __name__ == "__main__":
    main()
