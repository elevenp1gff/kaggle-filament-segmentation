"""Train with an ImageNet-pretrained encoder via segmentation_models_pytorch.

Why this is the right last move: everything so far trained from random init on 707
images, and 20+ decoder/schedule/augmentation variants all landed on 0.32. The
honest-ceiling analysis showed the probabilities themselves are the limit (recall
83-87% but precision only 63%), and precision is exactly what a pretrained backbone
improves first - low-level oriented-edge filters transfer from ImageNet even to a
domain as different as H-alpha solar images, because filaments are oriented dark
ridges.

The data pipeline, loss and decode are unchanged from the proven recipe so the
comparison isolates the encoder. Warm-start the backbone, let the head learn from
scratch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

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
from train_res import (CompositeLoss, augment, build_targets, decode_cc,  # noqa: E402
                       normalize_u8, predict_image, resolve_device)


class Source:
    """Same sampling policy as the proven PatchSource, kept deliberately simple."""

    def __init__(self, entries, size, cache_index, pos_prob=0.8, aug=True):
        self.entries = entries
        self.size = size
        self.paths = {e["file"]: cache_index["train/%s" % e["file"]]
                      for e in entries if "train/%s" % e["file"] in cache_index}
        self.rng = random.Random(1234)
        self.mem = {}
        self.pos_prob = pos_prob
        self.aug = aug

    def _arr(self, e):
        a = self.mem.get(e["file"])
        if a is None:
            a = np.load(self.paths[e["file"]])
            if len(self.mem) > 80:
                self.mem.clear()
            self.mem[e["file"]] = a
        return a

    def get(self, mode="train"):
        e = self.rng.choice(self.entries)
        x0, y0 = D.choose_window(e, self.rng, self.size, mode, self.pos_prob)
        arr = self._arr(e)
        patch = arr[y0:y0 + self.size, x0:x0 + self.size]
        if patch.shape[0] != self.size or patch.shape[1] != self.size:
            patch = cv2.copyMakeBorder(patch, 0, self.size - patch.shape[0],
                                       0, self.size - patch.shape[1],
                                       cv2.BORDER_REFLECT)
        insts = []
        for a in e["anns"]:
            bx, by, bw, bh = a["bbox"]
            if (bx + bw <= x0 or bx >= x0 + self.size or
                    by + bh <= y0 or by >= y0 + self.size):
                continue
            m = D.polygons_to_window_mask(a["segmentation"], x0, y0,
                                          self.size, self.size)
            if m.sum() > 0:
                insts.append(m)
        if self.aug and mode == "train":
            patch, insts = augment(patch, insts, self.rng, elastic=False)
        return normalize_u8(patch), insts


def build_model(encoder, weights, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=encoder, encoder_weights=weights,
                     in_channels=1, classes=2, activation=None)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--val-n", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--cache-dir", default="<CACHE_DIR>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    P("=== smp start %s encoder=%s weights=%s size=%d lr=%.1e"
      % (time.strftime("%F %T"), args.encoder, args.weights, args.size, args.lr))

    cache_index = json.load(open(os.path.join(args.cache_dir, "index.json")))
    idx = D.attach_paths(D.build_index(), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val, train = idx[:args.val_n], idx[args.val_n:]
    for e in train:
        e["split"] = "train"
    train = [e for e in train if "train/%s" % e["file"] in cache_index]
    P("train %d / val %d" % (len(train), len(val)))

    src = Source(train, args.size, cache_index, pos_prob=0.8, aug=True)
    model = build_model(args.encoder, args.weights, device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    P("params %.2fM" % n_par)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total, pct_start=0.1)
    crit = CompositeLoss()

    best = -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ls = []
        for _ in range(args.steps):
            xs, ys = [], []
            for _ in range(args.bs):
                p, insts = src.get("train")
                xs.append(torch.from_numpy(p[None].copy()))
                ys.append(torch.from_numpy(build_targets(insts, args.size)))
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(x)
            loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
        P("ep %3d/%d loss %.4f lr %.2e %.0fs"
          % (ep + 1, args.epochs, float(np.mean(ls)), sched.get_last_lr()[0],
             time.time() - t0))

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            gts_all, pr_all = [], []
            with torch.no_grad():
                for e in val:
                    arr = np.load(cache_index["train/%s" % e["file"]])
                    prob = predict_image(model, arr, args.size,
                                         int(args.size * 0.75), device)
                    g = []
                    for ann in e["anns"]:
                        m = D.polygons_to_window_mask(ann["segmentation"],
                                                      0, 0, 2048, 2048)
                        if m.sum() > 0:
                            g.append(m)
                    gts_all.append(g)
                    pr_all.append(decode_cc(prob[0], thr=0.25, min_area=400))
            r = panoptic_quality(gts_all, pr_all)
            P("   VAL PQ=%.4f tp=%d fp=%d fn=%d"
              % (r["pq"], r["tp"], r["fp"], r["fn"]))
            if r["pq"] > best:
                best = r["pq"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "pq": r["pq"]}, os.path.join(args.out, "best.pt"))
                P("   saved best PQ=%.4f" % r["pq"])
    P("=== done best %.4f in %.1f min ===" % (best, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
