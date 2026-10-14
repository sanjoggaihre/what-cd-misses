#!/usr/bin/env python
"""
linear_probe.py -- does the density-aware loss improve the ENCODER, or only
the decoder?

Every result so far measures decoder output. But the encoder is what the
downstream forecasting stage consumes, and a decoder-side loss change may
leave it untouched. This probe answers that directly.

METHOD
------
For each checkpoint: freeze the encoder, extract token features for every
frame at mask_ratio 0 (no masking -- we want the representation, not the
inpainting task), pool the 512 tokens, and fit a ridge regression predicting
the frame's growth day. Train on the training plants, select the ridge
penalty on the validation plants, report on the held-out test plants.

Splits are plant-wise, so test plants are entirely unseen.

WHAT A DIFFERENCE MEANS
-----------------------
Higher R^2 means the frozen representation carries more growth-stage
information. Since every encoder sees identical data and differs only in the
loss its decoder was trained with, a difference is attributable to that loss.

CAVEAT worth stating in any writeup: growth day correlates with plant size,
which is partly recoverable from local patch geometry regardless of
representation quality. The absolute R^2 is therefore not a measure of how
"good" the representation is in general. The comparison across encoders is
the meaningful quantity, since all of them have access to the same cue.

Usage:
    python linear_probe.py --runs runs.json --out probe_results.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from utils.config import cfg_from_yaml_file          # noqa: E402
from tools import builder                            # noqa: E402
from datasets import build_dataset_from_cfg          # noqa: E402


# (label, config path, checkpoint path). Edit to match your layout.
DEFAULT_RUNS = [
    ("Point-MAE (local CD)",
     "cfgs/pretrain_pepper_baseline.yaml",
     "experiments/pretrain_pepper_baseline/cfgs/PointMAE_baseline_exp1/ckpt-epoch-300.pth"),
    ("CD-L1 (assembled)",
     "cfgs/dloss_w000.yaml",
     "experiments/dloss_w000/cfgs/dloss_w000/ckpt-epoch-300.pth"),
    ("+ density w=0.002",
     "cfgs/dloss_w002.yaml",
     "experiments/dloss_w002/cfgs/dloss_w002_v2/ckpt-epoch-300.pth"),
    ("+ density w=0.005",
     "cfgs/pretrain_pepper_baseline_dloss.yaml",
     "experiments/pretrain_pepper_baseline_dloss/cfgs/baseline_dloss/ckpt-epoch-300.pth"),
    ("+ density w=0.010",
     "cfgs/dloss_w010.yaml",
     "experiments/dloss_w010/cfgs/dloss_w010_v2/ckpt-epoch-300.pth"),
]

ALPHAS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]


def day_from_id(frame_id):
    """Frame ids look like plant_03_0090; the trailing digits are the day."""
    m = re.search(r"_(\d+)$", str(frame_id))
    if m is None:
        raise ValueError(f"cannot parse a day from frame id {frame_id!r}")
    return int(m.group(1))


@torch.no_grad()
def extract(model, ds, device="cuda", batch=8):
    """Pooled token features and growth day for every frame in ds.

    Mask ratio is set to 0 so the encoder sees the complete token set. We want
    to probe the representation, not the model's ability to inpaint.
    """
    feats, days, ids = [], [], []
    buf_pts, buf_meta = [], []

    def flush():
        if not buf_pts:
            return
        pts = torch.stack(buf_pts).to(device)
        neighborhood, center = model.group_divider(pts)
        x_vis, mask = model.MAE_encoder(neighborhood, center)
        if mask.sum() > 0:
            raise RuntimeError(
                "encoder is still masking -- mask_ratio override did not take")
        # (B, G, C) -> concat of mean and max pooling over tokens.
        f = torch.cat([x_vis.mean(1), x_vis.max(1).values], dim=1)
        feats.append(f.cpu().numpy())
        for fid in buf_meta:
            ids.append(fid)
            days.append(day_from_id(fid))
        buf_pts.clear()
        buf_meta.clear()

    for i in range(len(ds)):
        np.random.seed(1000+i)
        item = ds[i]
        data = item[2]
        pts = data["points"] if isinstance(data, dict) else data
        buf_pts.append(pts)
        buf_meta.append(item[1])
        if len(buf_pts) == batch:
            flush()
    flush()

    return np.concatenate(feats), np.asarray(days, dtype=float), ids


def probe(Xtr, ytr, Xva, yva, Xte, yte, test_ids=None):
    """Ridge with the penalty chosen on validation, reported on test."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    best, best_a = -np.inf, None
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr, ytr)
        s = r.score(Xva, yva)
        if s > best:
            best, best_a = s, a

    model = Ridge(alpha=best_a).fit(Xtr, ytr)
    pred = model.predict(Xte)
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    if test_ids is not None:
        plants = np.array([re.match(r'(plant_\d+)', str(i)).group(1)
                           for i in test_ids])
        print("   per-plant test R2 (4 plants; frames within a plant are "
              "correlated, so this is the effective sample size):")
        for p in np.unique(plants):
            m = plants == p
            r2 = 1 - ((yte[m] - pred[m]) ** 2).sum() / \
                     ((yte[m] - yte[m].mean()) ** 2).sum()
            print(f"     {p}: R2 {r2:>7.3f}  n={m.sum()}")
    return {
        "alpha": best_a,
        "val_r2": float(best),
        "test_r2": 1.0 - ss_res / ss_tot,
        "test_mae_days": float(np.abs(yte - pred).mean()),
        "test_rmse_days": float(np.sqrt(((yte - pred) ** 2).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None,
                    help="optional JSON list of [label, config, ckpt]")
    ap.add_argument("--out", default="probe_results.json")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    runs = DEFAULT_RUNS
    if args.runs:
        with open(args.runs) as fh:
            runs = [tuple(r) for r in json.load(fh)]

    results = {}
    for label, cfg_path, ckpt_path in runs:
        if not os.path.exists(ckpt_path):
            print(f"!! skipping {label}: {ckpt_path} not found")
            continue
        print(f"\n=== {label} ===")

        cfg = cfg_from_yaml_file(cfg_path)
        model = builder.model_builder(cfg.model)
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("base_model", state.get("model", state))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"   {len(missing)} missing keys (decoder-side is expected)")
        model = model.cuda().eval()
        model.MAE_encoder.mask_ratio = 0.0

        split_feats = {}
        for split in ("train", "val", "test"):
            dcfg = getattr(cfg.dataset, split)
            ds = build_dataset_from_cfg(dcfg._base_, dcfg.others)
            X, y, ids = extract(model, ds, batch=args.batch)
            split_feats[split] = (X, y, ids)
            print(f"   {split:<6} {X.shape[0]:>4} frames, dim {X.shape[1]}, "
                  f"days {y.min():.0f}-{y.max():.0f}")

        Xtr, ytr, _ = split_feats["train"]
        Xva, yva, _ = split_feats["val"]
        Xte, yte, te_ids = split_feats["test"]
        r = probe(Xtr, ytr, Xva, yva, Xte, yte, test_ids=te_ids)
        results[label] = r
        print(f"   test R2 {r['test_r2']:.4f}   MAE {r['test_mae_days']:.2f} d"
              f"   (alpha {r['alpha']:g}, val R2 {r['val_r2']:.4f})")

        del model
        torch.cuda.empty_cache()

    # A constant predictor is the floor: if every encoder lands near it, the
    # representation carries no growth-stage information at all.
    print(f"\n{'Encoder':<26}{'test R2':>10}{'MAE (days)':>13}{'RMSE':>9}")
    print("-" * 58)
    for label, r in results.items():
        print(f"{label:<26}{r['test_r2']:>10.4f}"
              f"{r['test_mae_days']:>13.2f}{r['test_rmse_days']:>9.2f}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()