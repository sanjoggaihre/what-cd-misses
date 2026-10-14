#seg_downstream.py

"""
Downstream test: stem/leaf semantic segmentation of reconstructed plants.

A DGCNN segmenter is trained on ground-truth scans only (labelled frames of the
subset-1 training plants; plant_05 for validation), frozen, and applied to each
model's reconstructions of the labelled test frames (plant_03 up to day 62, plant_11).
A faithful reconstruction should segment about as well as the ground truth.

Reported per model, pooled over frames (IoU per class, then their mean):
  output    IoU on the model's output points; reference label = nearest GT point's label
  inferred  same, restricted to generated points. Point-MAE copies ~40% of its output
            from the input; copied points coincide exactly with GT points and are dropped.
            For the coordination decoder (copies nothing) inferred == output.
  on_gt     predictions transferred back to the GT points (each GT point takes the class
            of its nearest output point): how well the plant's real stem/leaf split is
            recovered from the reconstruction
  gated     as on_gt, but a GT point takes a class only if an output point lies within
            tau (mean GT nearest-neighbour spacing, the recall threshold); uncovered GT
            points count as wrong. Links segmentation to geometric coverage. cov_stem /
            cov_leaf report the covered fractions per class.
            Note: on the complete cloud Point-MAE's copied input covers its own region for
            free, so this comparison is conservative -- biased toward the baseline.
Row "ground truth" applies the segmenter to the GT subsample itself: the ceiling.

Needs per_organ_recall.py in the same directory (frame matching).
Pure PyTorch: no compiled extensions.

Usage:
  python seg_downstream.py --smoke        # few-minute check that every step runs
  python seg_downstream.py --train        # cache data (first run only), train, evaluate
  python seg_downstream.py --models baseline w005 use_refine_v4 <repulsion_dir>   # evaluate only
  (add --ckpt seg_dgcnn_s1.pth --seed 1 --out seg_s1.csv to evaluate a saved seed without retraining)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import per_organ_recall as por  # noqa: E402

TRAIN_PLANTS = ["plant_01", "plant_02", "plant_04", "plant_06",
                "plant_07", "plant_08", "plant_09", "plant_10"]
VAL_PLANTS = ["plant_05"]
TEST_PLANTS = ["plant_03", "plant_11"]
COPY_TOL = 1e-6      # output point this close to a GT point = copied input (normalised units)


# ------------------------------------------------------------------ data
def load_labelled_frames(plants, cache, limit=None):
    """[(raw xyz float32, semantic label int8)] for every frame with semantic labels.
    Labels: 0 stem, 1 leaf, -1 unlabelled (ignored)."""
    os.makedirs(cache, exist_ok=True)
    frames = []
    for plant in plants:
        pdir = por.plant_dir(plant)
        sems = sorted(glob.glob(os.path.join(pdir, "labels", "semantic", "*_semantic.txt")))
        for sem_path in sems[:limit]:
            stem_name = os.path.basename(sem_path)[:-len("_semantic.txt")]
            cf = os.path.join(cache, f"{plant}_{stem_name}.npz")
            if not os.path.exists(cf):
                raw = np.load(os.path.join(pdir, "frames", stem_name + ".npy"))[:, :3]
                lab = pd.read_csv(sem_path, sep=r"\s+", header=None).to_numpy()
                d, idx = cKDTree(raw).query(lab[:, :3])
                ok = d < por.MATCH_TOL
                y = np.full(len(raw), -1, np.int8)
                y[idx[ok]] = np.rint(lab[ok, -1]).astype(np.int8)   # label = last column
                np.savez(cf, xyz=raw.astype(np.float32), y=y)
            z = np.load(cf)
            frames.append((z["xyz"], z["y"]))
        print(f"  {plant}: {len(frames)} labelled frames loaded so far", flush=True)
    return frames


def rotation(rng, max_deg=10.0):
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    t = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def sample(xyz, y, rng, scale, npts, aug):
    """Uniform random subsample, like the evaluation GT; normalise by the global scale
    and centre on the cloud's own centroid (the evaluation centres on the GT centroid)."""
    ch = rng.choice(len(xyz), npts, replace=len(xyz) < npts)
    p = xyz[ch].astype(np.float64) / scale
    p = p - p.mean(0)
    if aug:
        p = (p @ rotation(rng).T) * rng.uniform(0.85, 1.15) + rng.normal(0, 0.002, p.shape)
    return p.astype(np.float32), y[ch].astype(np.int64)


# ------------------------------------------------------------------ model (DGCNN, Wang et al. 2019)
@torch.no_grad()
def knn(x, k, chunk=2048):
    """x: B x C x N -> indices B x N x k (chunked so 16k-point clouds fit in memory)."""
    xt = x.transpose(1, 2)
    sq = (xt ** 2).sum(-1)
    out = []
    for s in range(0, xt.shape[1], chunk):
        q = xt[:, s:s + chunk]
        d = sq[:, s:s + chunk, None] - 2 * q @ xt.transpose(1, 2) + sq[:, None, :]
        out.append(d.topk(k, dim=-1, largest=False).indices)
    return torch.cat(out, 1)


def edge_features(x, k):
    B, C, N = x.shape
    idx = knn(x, k) + torch.arange(B, device=x.device).view(-1, 1, 1) * N
    xt = x.transpose(1, 2).reshape(B * N, C)
    nb = xt[idx.reshape(-1)].view(B, N, k, C)
    ctr = x.transpose(1, 2).unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([nb - ctr, ctr], -1).permute(0, 3, 1, 2).contiguous()   # B x 2C x N x k


class DGCNNSeg(nn.Module):
    def __init__(self, k=20, n_cls=2):
        super().__init__()
        self.k = k

        def b2(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))

        def b1(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 1, bias=False), nn.BatchNorm1d(o), nn.LeakyReLU(0.2))

        self.e1 = nn.Sequential(b2(6, 64), b2(64, 64))
        self.e2 = nn.Sequential(b2(128, 64), b2(64, 64))
        self.e3 = b2(128, 64)
        self.glob = b1(192, 1024)
        self.head = nn.Sequential(b1(1024 + 192, 256), nn.Dropout(0.5), b1(256, 128),
                                  nn.Conv1d(128, n_cls, 1))

    def forward(self, x):                                   # x: B x 3 x N
        n = x.shape[2]
        x1 = self.e1(edge_features(x, self.k)).max(-1)[0]
        x2 = self.e2(edge_features(x1, self.k)).max(-1)[0]
        x3 = self.e3(edge_features(x2, self.k)).max(-1)[0]
        f = torch.cat([x1, x2, x3], 1)
        g = self.glob(f).max(-1, keepdim=True)[0].expand(-1, -1, n)
        return self.head(torch.cat([g, f], 1))              # B x n_cls x N


@torch.no_grad()
def predict(net, pts, device):
    net.eval()
    x = torch.from_numpy(np.ascontiguousarray(pts.T[None], dtype=np.float32)).to(device)
    return net(x).argmax(1)[0].cpu().numpy()


# ------------------------------------------------------------------ metrics
def new_acc():
    return np.zeros((2, 2))                                 # [class, (intersection, union)]


def accumulate(acc, pred, ref):
    m = ref >= 0
    p, r = pred[m], ref[m]
    for c in (0, 1):
        acc[c, 0] += np.sum((p == c) & (r == c))
        acc[c, 1] += np.sum((p == c) | (r == c))


def miou(acc):
    iou = acc[:, 0] / np.maximum(acc[:, 1], 1)
    return float(iou.mean()), float(iou[0]), float(iou[1])     # mean, stem, leaf


# ------------------------------------------------------------------ train
def val_miou(net, frames, scale, npts, device):
    rng = np.random.default_rng(123)
    acc = new_acc()
    for xyz, y in frames:
        p, t = sample(xyz, y, rng, scale, npts, aug=False)
        accumulate(acc, predict(net, p, device), t)
    return miou(acc)


def train(args, scale, device):
    limit = 4 if args.smoke else None
    print("loading training frames (first run caches them to disk)", flush=True)
    tr = load_labelled_frames(TRAIN_PLANTS[:1] if args.smoke else TRAIN_PLANTS, args.cache, limit)
    va = load_labelled_frames(VAL_PLANTS, args.cache, 2 if args.smoke else None)
    counts = np.bincount(np.concatenate([y[y >= 0] for _, y in tr]), minlength=2).astype(float)
    print(f"train frames {len(tr)}, val frames {len(va)}, stem share {counts[0] / counts.sum():.3f}")
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)

    net = DGCNNSeg(k=args.k).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.epochs
    steps = 2 if args.smoke else max(len(tr) // args.batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    rng = np.random.default_rng(args.seed)
    best = -1.0
    for ep in range(epochs):
        net.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(len(tr))
        for s in range(steps):
            ids = order[(s * args.batch + np.arange(args.batch)) % len(tr)]
            batch = [sample(*tr[i], rng, scale, args.npts, aug=True) for i in ids]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).transpose(1, 2).contiguous().to(device)
            t = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
            loss = F.cross_entropy(net(x), t, weight=w, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f"epoch {ep + 1}/{epochs} loss {tot / steps:.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            m, s_iou, l_iou = val_miou(net, va, scale, args.npts, device)
            msg += f" | val mIoU {m:.3f} (stem {s_iou:.3f}, leaf {l_iou:.3f})"
            if m > best:
                best = m
                torch.save(net.state_dict(), args.ckpt)
                msg += "  saved"
        print(msg, flush=True)


# ------------------------------------------------------------------ evaluate reconstructions
def gt_semantic(gt_path, scale, allow_temporal=False):
    """GT points of a saved *_gt.ply and their stem/leaf labels (None if unlabelled day).
    Subset-1 plants use instance labels. With allow_temporal, subset-3 plants use temporal
    IDs with the same convention (0 and -2 = stem, >0 = leaf) -- only valid once the
    subset-3 label format has been checked."""
    plant, num = por.GT_RE.search(os.path.basename(gt_path)).groups()
    num = int(num)
    pdir = por.plant_dir(plant)
    gt = por.read_ply_xyz(gt_path)
    ok_sources = ("instance", "temporal") if allow_temporal else ("instance",)
    for day in dict.fromkeys((num, 2 * num)):
        fr = por.raw_frame(pdir, day)
        if fr is None or fr[2] is None or fr[3] not in ok_sources:
            continue
        idx = por.align_to_frame(gt, fr[0], fr[1], scale)
        if idx is not None:
            inst = fr[2][idx]                       # -2 already merged into stem (0)
            return gt, np.where(inst == 0, 0, np.where(inst > 0, 1, -1)), day
    return gt, None, None


BASES = ("output", "inferred", "on_gt", "gated")
UNCOVERED = -9      # class given to GT points with no output point within tau: always wrong


def evaluate_models(args, scale, device):
    net = DGCNNSeg(k=args.k).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    test_plants = TEST_PLANTS + (["plant_28", "plant_29"] if args.include_temporal else [])
    rows, ceiling, seen, frame_rows = [], new_acc(), set(), []
    ceil_frame = {}
    for model in args.models:
        paths = sorted(p for p in glob.glob(os.path.join(por.CDF, model, "*_gt.ply"))
                       if os.path.basename(p).rsplit("_", 2)[0] in test_plants)
        if args.smoke:
            paths = paths[:1]
        acc = {k: new_acc() for k in BASES}
        cov = np.zeros((2, 2))                      # [class, (covered, total)]
        copied, used = [], 0
        for gp in paths:
            gt, sem, _ = gt_semantic(gp, scale, args.include_temporal)
            if sem is None:
                continue
            used += 1
            pred = por.read_ply_xyz(gp.replace("_gt.ply", "_rec.ply"))
            c = gt.mean(0)
            g = (gt - c).astype(np.float32)
            q = (pred - c).astype(np.float32)
            key = os.path.basename(gp)[:-len("_gt.ply")]
            if key not in seen:                     # GT subsample is identical across models
                fa = new_acc()
                accumulate(fa, predict(net, g, device), sem)
                ceiling += fa
                ceil_frame[key] = miou(fa)
                seen.add(key)
            tau = cKDTree(g).query(g, k=2)[0][:, 1].mean()   # same tau as the recall metric
            pp = predict(net, q, device)
            d, j = cKDTree(g).query(q)
            ref = sem[j]
            cp = d < COPY_TOL
            copied.append(cp.mean())
            dg, jg = cKDTree(q).query(g)            # GT -> nearest output point
            covered = dg <= tau
            fr = {k: new_acc() for k in BASES}
            accumulate(fr["output"], pp, ref)
            accumulate(fr["inferred"], pp[~cp], ref[~cp])
            accumulate(fr["on_gt"], pp[jg], sem)
            accumulate(fr["gated"], np.where(covered, pp[jg], UNCOVERED), sem)
            frow = dict(model=model, frame=key, plant=key.rsplit("_", 1)[0], seed=args.seed,
                        gt_mIoU=ceil_frame[key][0], gt_stem=ceil_frame[key][1])
            for k in BASES:
                acc[k] += fr[k]
                m_, s_, l_ = miou(fr[k])
                frow.update({f"{k}_mIoU": m_, f"{k}_stem": s_, f"{k}_leaf": l_})
            lab = sem >= 0
            for cl, nm in ((0, "stem"), (1, "leaf")):
                mk = sem == cl
                cov[cl] += (covered[mk].sum(), mk.sum())
                frow[f"cov_{nm}"] = covered[mk].mean() if mk.any() else np.nan
            frow["cov_all"] = covered[lab].mean()
            frame_rows.append(frow)
        row = dict(model=model, frames=used, copied_share=np.mean(copied) if copied else np.nan)
        for k, a in acc.items():
            m, s_iou, l_iou = miou(a)
            row.update({f"{k}_mIoU": m, f"{k}_stem": s_iou, f"{k}_leaf": l_iou})
        row.update(cov_stem=cov[0, 0] / max(cov[0, 1], 1), cov_leaf=cov[1, 0] / max(cov[1, 1], 1))
        rows.append(row)
    m, s_iou, l_iou = miou(ceiling)
    rows.insert(0, dict(model="ground truth", frames=len(seen), copied_share=np.nan,
                        output_mIoU=m, output_stem=s_iou, output_leaf=l_iou,
                        gated_mIoU=m, gated_stem=s_iou, gated_leaf=l_iou,
                        cov_stem=1.0, cov_leaf=1.0))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(frame_rows).to_csv(args.out.replace(".csv", "_frames.csv"), index=False)
    show = ["model", "frames", "copied_share", "inferred_mIoU", "inferred_stem",
            "on_gt_mIoU", "gated_mIoU", "gated_stem", "gated_leaf", "cov_stem", "cov_leaf"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[show].round(3).to_string(index=False))
    print(f"\nsaved {args.out} and per-frame {args.out.replace('.csv', '_frames.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train even if a checkpoint exists")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to check the setup")
    ap.add_argument("--models", nargs="+", default=["baseline", "w005", "use_refine_v4", "dcd_a50_l05", "dcd_a50", "rep_h007_w05", "rep_h007_w1"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--npts", type=int, default=16384)
    ap.add_argument("--ckpt", default="seg_dgcnn.pth")
    ap.add_argument("--cache", default="seg_cache")
    ap.add_argument("--out", default="seg_downstream.csv")
    ap.add_argument("--seed", type=int, default=0, help="segmenter training seed")
    ap.add_argument("--include_temporal", action="store_true",
                    help="also evaluate plants 28/29 via temporal labels (check their format first)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    with open(os.path.join(por.DATA, "scale.json")) as f:
        scale = json.load(f)["global_scale"]
    if args.smoke:
        args.ckpt, args.out = "seg_smoke.pth", "seg_smoke.csv"
    if args.train or args.smoke or not os.path.exists(args.ckpt):
        train(args, scale, device)
    evaluate_models(args, scale, device)


if __name__ == "__main__":
    main()