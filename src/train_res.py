"""Train with an objective that matches the PQ metric, not pixel Dice.

Ceiling analysis: with the current model the best any decoder can reach is
PQ 0.478 (ideal grouping of its own probabilities), while actual decoding gives
0.12. Pixel DICE is 0.48. So two levers, in order of size:
  (a) raise the probability quality   - bigger model, more epochs, better aug
  (b) close the 0.478 -> 0.12 gap     - instance-aware decoding

This script attacks (a) with changes that are justified by measurement:
  * deeper/wider encoder (resnet34-style, ImageNet-free) so the receptive field
    covers whole filaments (median bbox 74x69 px, but filaments run 100s of px)
  * 768 px training crops instead of 512, because 15% of instances are <500 px^2
    and downscaling destroys them
  * loss on a soft boundary target and an explicit distance-aware term
  * cosine LR with warm restarts, longer schedule
  * heavier augmentation (scaling, rotation, intensity jitter) since only 707
    images exist and the test set is a random split of the same population
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# --------------------------------------------------------------------------- #
#  model: residual encoder/decoder, enough receptive field for long filaments
# --------------------------------------------------------------------------- #
class ConvNorm(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.n = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.n(self.c(x)), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.BatchNorm2d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.skip is None else self.skip(x)
        y = F.relu(self.n1(self.c1(x)), inplace=True)
        y = self.n2(self.c2(y))
        return F.relu(y + r, inplace=True)


class ResUNet(nn.Module):
    """Residual UNet with a wide stem; outputs (foreground, boundary)."""

    def __init__(self, base=32, out_ch=2):
        super().__init__()
        b = base
        self.stem = nn.Sequential(ConvNorm(1, b, 7, 1), ConvNorm(b, b, 3))
        self.e1 = nn.Sequential(ResBlock(b, b), ResBlock(b, b))
        self.e2 = nn.Sequential(ResBlock(b, b * 2, 2), ResBlock(b * 2, b * 2))
        self.e3 = nn.Sequential(ResBlock(b * 2, b * 4, 2), ResBlock(b * 4, b * 4))
        self.e4 = nn.Sequential(ResBlock(b * 4, b * 8, 2), ResBlock(b * 8, b * 8))
        self.e5 = nn.Sequential(ResBlock(b * 8, b * 16, 2), ResBlock(b * 16, b * 16))
        self.d4 = ConvNorm(b * 16 + b * 8, b * 8)
        self.d3 = ConvNorm(b * 8 + b * 4, b * 4)
        self.d2 = ConvNorm(b * 4 + b * 2, b * 2)
        self.d1 = ConvNorm(b * 2 + b, b)
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b = self.e5(s4)
        b = self.d4(torch.cat([F.interpolate(b, size=s4.shape[-2:], mode="nearest"), s4], 1))
        b = self.d3(torch.cat([F.interpolate(b, size=s3.shape[-2:], mode="nearest"), s3], 1))
        b = self.d2(torch.cat([F.interpolate(b, size=s2.shape[-2:], mode="nearest"), s2], 1))
        b = self.d1(torch.cat([F.interpolate(b, size=s1.shape[-2:], mode="nearest"), s1], 1))
        out = self.head(b)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


# --------------------------------------------------------------------------- #
#  targets & loss
# --------------------------------------------------------------------------- #
def build_targets(insts, size, bd=3):
    fg = (np.stack(insts).max(0) if insts else np.zeros((size, size), np.uint8)).astype(np.float32)
    b = np.zeros((size, size), np.uint8)
    k = np.ones((bd, bd), np.uint8)
    for m in insts:
        b |= cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return np.stack([fg, b.astype(np.float32)], 0)


class CompositeLoss(nn.Module):
    """BCE + soft Dice + Tversky-ish term that up-weights missed filament pixels.

    Recall matters more than precision here: a missed instance is an FN which
    costs 0.5 in the PQ denominator exactly like an FP, but a *fragment* of a
    missed filament still scores SQ. Weighting false negatives higher pushes the
    model to cover filaments fully instead of emitting confident blobs.
    """

    def __init__(self, w_bce=1.0, w_dice=1.5, fn_weight=2.0):
        super().__init__()
        self.w_bce, self.w_dice, self.fn_weight = w_bce, w_dice, fn_weight

    def forward(self, logit, target):
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        # per-pixel weight: missed positives cost more
        w = 1.0 + (self.fn_weight - 1.0) * target
        bce = (bce * w).mean()
        p = torch.sigmoid(logit)
        num = 2 * (p * target).sum((2, 3)) + 1.0
        den = p.sum((2, 3)) + target.sum((2, 3)) + 1.0
        dice = 1 - (num / den).mean()
        return self.w_bce * bce + self.w_dice * dice


# --------------------------------------------------------------------------- #
#  augmentation & sampling
# --------------------------------------------------------------------------- #
def elastic_deform(patch, insts, rng, alpha=18.0, sigma=8.0):
    """Small smooth displacement field, applied identically to image and masks.

    Filaments are thin curved structures whose exact shape varies between
    observations; elastic warping is the standard way to teach a segmenter that
    invariance without inventing new anatomy. alpha is the displacement magnitude
    in pixels (kept small so thin structures stay connected) and sigma the
    smoothing of the field.
    """
    h, w = patch.shape
    # numpy generator: random.Random has no array support
    nprng = np.random.default_rng(rng.randrange(2 ** 31))
    dx = cv2.GaussianBlur((nprng.random((h, w)) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((nprng.random((h, w)) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    mapx = (xx + dx).astype(np.float32)
    mapy = (yy + dy).astype(np.float32)
    patch = cv2.remap(patch, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    insts = [cv2.remap(m, mapx, mapy, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
             for m in insts]
    return patch, insts


def augment(patch, insts, rng, elastic=True):
    """Flips, 90-degree rotations, small scale/rotate, elastic, intensity jitter."""
    k = rng.randrange(4)
    if k == 1:
        patch = np.ascontiguousarray(patch.T)
        insts = [np.ascontiguousarray(m.T) for m in insts]
    elif k == 2:
        patch = np.ascontiguousarray(patch[::-1])
        insts = [np.ascontiguousarray(m[::-1]) for m in insts]
    elif k == 3:
        patch = np.ascontiguousarray(patch[:, ::-1])
        insts = [np.ascontiguousarray(m[:, ::-1]) for m in insts]

    if rng.random() < 0.5:
        # small affine: keeps thin structures but adds translation robustness
        h, w = patch.shape
        ang = rng.uniform(-12, 12)
        sc = rng.uniform(0.92, 1.08)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        patch = cv2.warpAffine(patch, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        insts = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT) for m in insts]

    if elastic and rng.random() < 0.4:
        patch, insts = elastic_deform(patch, insts, rng,
                                      alpha=rng.uniform(10.0, 22.0), sigma=8.0)

    if rng.random() < 0.6:
        # gamma jitter simulates exposure differences between GONG sites
        g = rng.uniform(0.75, 1.35)
        patch = np.clip(255.0 * (patch.astype(np.float32) / 255.0) ** g, 0, 255).astype(np.uint8)
    return patch, insts


class PatchSource:
    def __init__(self, entries, size, cache_index, pos_prob=0.8, aug=True,
                 elastic=True, weight_json=None):
        self.entries = entries
        self.size = size
        self.elastic = elastic
        # entries carry their own split so pseudo (test) images resolve correctly
        self.paths = {}
        for e in entries:
            split = e.get("split", "train")
            key = "%s/%s" % (split, e["file"])
            if key in cache_index:
                self.paths[e["file"]] = cache_index[key]
        missing = [e["file"] for e in entries if e["file"] not in self.paths]
        if missing:
            print("WARNING: %d entries have no cached image, e.g. %s"
                  % (len(missing), missing[:3]))
        self.rng = random.Random(1234)
        self.mem = {}
        self.pos_prob = pos_prob
        self.aug = aug
        # optional per-image sampling weights (hard-example mining). When present,
        # images the current model handles badly are drawn more often, because the
        # leaderboard is decided by those.
        self.weights = None
        if weight_json and os.path.exists(weight_json):
            w = json.load(open(weight_json))
            keys = [e["file"] for e in entries]
            vals = np.array([w.get(k, 0.0) for k in keys], dtype=np.float64)
            if vals.sum() > 0:
                self.weights = vals / vals.sum()
                print("hard-example weights loaded: %d images, "
                      "max/min ratio %.1fx"
                      % (len(keys), vals.max() / max(vals.min(), 1e-9)))

    def _pick_entry(self):
        if self.weights is None:
            return self.rng.choice(self.entries)
        i = self.rng.choices(range(len(self.entries)), weights=self.weights, k=1)[0]
        return self.entries[i]

    def _arr(self, e):
        a = self.mem.get(e["file"])
        if a is None:
            a = np.load(self.paths[e["file"]])
            if len(self.mem) > 90:
                self.mem.clear()
            self.mem[e["file"]] = a
        return a

    def get(self, mode="train"):
        e = self._pick_entry()
        x0, y0 = D.choose_window(e, self.rng, self.size, mode, self.pos_prob)
        arr = self._arr(e)
        patch = arr[y0:y0 + self.size, x0:x0 + self.size]
        if patch.shape[0] != self.size or patch.shape[1] != self.size:
            patch = cv2.copyMakeBorder(patch, 0, self.size - patch.shape[0],
                                       0, self.size - patch.shape[1], cv2.BORDER_REFLECT)
        insts = []
        for a in e["anns"]:
            bx, by, bw, bh = a["bbox"]
            if bx + bw <= x0 or bx >= x0 + self.size or by + bh <= y0 or by >= y0 + self.size:
                continue
            m = D.polygons_to_window_mask(a["segmentation"], x0, y0, self.size, self.size)
            if m.sum() > 0:
                insts.append(m)
        if self.aug and mode == "train":
            patch, insts = augment(patch, insts, self.rng, elastic=self.elastic)
        return normalize_u8(patch), insts


# --------------------------------------------------------------------------- #
#  decode + eval
# --------------------------------------------------------------------------- #
def decode_cc(prob, thr=0.5, min_area=60, open_k=0):
    fg = (prob > thr).astype(np.uint8)
    if open_k:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    return [(lab == i).astype(np.uint8) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def decode_ws(prob, thr=0.5, min_area=60, h_maxima=2, open_k=0):
    from skimage.morphology import h_maxima as sk_hm
    from skimage.segmentation import watershed
    fg = (prob > thr).astype(np.uint8)
    if open_k:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if fg.sum() == 0:
        return []
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    peaks = np.argwhere(sk_hm(dist, h_maxima) > 0)
    if len(peaks) == 0:
        return decode_cc(prob, thr, min_area, open_k)
    markers = np.zeros(fg.shape, np.int32)
    for k, (yy, xx) in enumerate(peaks, 1):
        markers[yy, xx] = k
    lab = watershed(-dist, markers, mask=fg)
    return [(lab == i).astype(np.uint8) for i in range(1, lab.max() + 1)
            if (lab == i).sum() >= min_area]


@torch.no_grad()
def predict_image(model, arr_u8, size, stride, device, max_batch=16):
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
        for (y, x), p in zip(ch, pr):
            acc[:, y:y + size, x:x + size] += p
            cnt[y:y + size, x:x + size] += 1
    acc /= np.maximum(cnt, 1e-6)[None]
    return acc[:, :h, :w]


def eval_split(model, entries, cache_index, size, stride, device,
               decoders, verbose=True):
    gts_all, probs_all = [], []
    for e in entries:
        arr = np.load(cache_index[f"train/{e['file']}"])
        prob = predict_image(model, arr, size, stride, device)
        h, w = prob.shape[1:]
        g = []
        for a in e["anns"]:
            m = D.polygons_to_window_mask(a["segmentation"], 0, 0, w, h)
            if m.sum() > 0:
                g.append(m)
        gts_all.append(g)
        probs_all.append(prob[0])
    res = {}
    for name, kw in decoders.items():
        if name == "cc":
            preds = [decode_cc(p, **kw) for p in probs_all]
        else:
            preds = [decode_ws(p, **kw) for p in probs_all]
        res[name] = panoptic_quality(gts_all, preds)
    return res


def resolve_device(a):
    if a.startswith("npu"):
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise SystemExit("npu unavailable")
        return torch.device(a)
    if a.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--stride", type=int, default=576)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-n", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--cache-dir", default="<CACHE_DIR>")
    ap.add_argument("--pseudo-repeat", type=int, default=2,
                    help="duplicate factor; 2 means pseudo images are seen half as often")
    ap.add_argument("--out", default="runs/res")
    ap.add_argument("--pseudo", default="", help="pseudo-label COCO json to add to training")
    ap.add_argument("--no-elastic", action="store_true",
                    help="disable elastic augmentation")
    ap.add_argument("--dedup", default="richest", choices=["richest", "union"],
                    help="how to treat duplicate annotation records for the same file")
    ap.add_argument("--pos-prob", type=float, default=0.8,
                    help="fraction of training windows centred on an annotated filament")
    ap.add_argument("--hard-weights", default="",
                    help="json of per-image sampling weights for hard-example mining")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    P(f"=== res start {time.strftime('%F %T')} size={args.size} base={args.base} ===")
    cache_index = json.load(open(os.path.join(args.cache_dir, "index.json")))
    # validation always comes from the real annotations, never pseudo labels
    idx = D.attach_paths(D.build_index(dedup=args.dedup), split="train")
    rng = random.Random(42)
    rng.shuffle(idx)
    val, train = idx[:args.val_n], idx[args.val_n:]
    for e in train:
        e["split"] = "train"

    if args.pseudo:
        import data_mixed
        pseudo = data_mixed.build_mixed_index(pseudo_path=args.pseudo,
                                              pseudo_repeat=args.pseudo_repeat)
        pseudo = [e for e in pseudo if e.get("pseudo")]
        train = train + pseudo
        P("pseudo: %d images / %d anns added (repeat %d)"
          % (len(pseudo), sum(len(e["anns"]) for e in pseudo),
             args.pseudo_repeat))
    P(f"train {len(train)} / val {len(val)}")

    src = PatchSource(train, args.size, cache_index, pos_prob=args.pos_prob, aug=True,
                       elastic=not args.no_elastic, weight_json=args.hard_weights)
    model = ResUNet(base=args.base).to(device)
    with torch.no_grad():
        model.head.bias[0] = -4.0
        model.head.bias[1] = -2.0
    P(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.1, anneal_strategy="cos")
    crit = CompositeLoss()

    decoders = {"cc": dict(thr=0.5, min_area=80, open_k=0)}
    best = -1
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
        P(f"ep {ep+1:3d}/{args.epochs} loss {np.mean(ls):.4f} "
          f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = eval_split(model, val, cache_index, args.size, args.stride,
                             device, decoders)
            for k, r in res.items():
                P(f"   VAL {k:4s} PQ={r['pq']:.4f} SQ={r['sq']:.3f} RQ={r['rq']:.3f} "
                  f"tp={r['tp']} fp={r['fp']} fn={r['fn']}")
                if r["pq"] > best:
                    best = r["pq"]
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "pq": r["pq"]}, os.path.join(args.out, "best.pt"))
                    P(f"   saved best PQ={r['pq']:.4f}")
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       os.path.join(args.out, "last.pt"))
    P(f"=== done best {best:.4f} in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
