#!/usr/bin/env python
"""
Figures for per-organ recall on Pepper-4D test plants with instance labels (03, 11).

Figure 1  per_organ_<frame>.png/.pdf  -- one test frame, one column per model
  top row:    GT points coloured by the recall of the organ (leaf / stem) they belong to
  bottom row: GT points the model missed (no prediction within tau) in red, covered in grey.
              This shows the holes directly, which a prediction-coloured render cannot.
  The view is computed from the data: vertical = the stem's principal axis,
  horizontal = the plant's widest direction. Identical for every panel.

Figure 2  per_leaf_recall_distribution.png/.pdf  -- from per_organ_recall.csv
  left:  cumulative distribution of per-leaf recall, one line per model
  right: paired scatter, each leaf's recall under two models (points above the
         diagonal improved), coloured by plant

Needs per_organ_recall.py in the same directory; reuses its frame matching.

Usage:
  python per_organ_figures.py                                  # plant_03 frame nearest day 40
  python per_organ_figures.py --frame plant_11_0020
  python per_organ_figures.py --models baseline w000 w005 w010 use_refine_v4 --flip
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import per_organ_recall as por  # noqa: E402

NAMES = {"baseline": "Point-MAE (CD)", "w000": "CD, assembled", "w002": "Ours (w=0.002)",
         "w005": "Ours (w=0.005)", "w010": "Ours (w=0.010)",
         "use_refine_v3": "Coordination v3", "use_refine_v4": "Coordination decoder"}


def name(m):
    return NAMES.get(m, m)


def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.join(out_dir, stem)}.png/.pdf")


def frame_data(model, frame, scale):
    """GT points, their organ labels, and per-point hit/miss for one model and frame."""
    gt_path = os.path.join(por.CDF, model, f"{frame}_gt.ply")
    gt = por.read_ply_xyz(gt_path)
    pred = por.read_ply_xyz(gt_path.replace("_gt.ply", "_rec.ply"))
    plant, num = frame.rsplit("_", 1)
    num = int(num)
    pdir = por.plant_dir(plant)
    for day in dict.fromkeys((num, 2 * num)):
        fr = por.raw_frame(pdir, day)
        if fr is None or fr[2] is None:
            continue
        idx = por.align_to_frame(gt, fr[0], fr[1], scale)
        if idx is not None:
            tau = cKDTree(gt).query(gt, k=2)[0][:, 1].mean()
            hit = cKDTree(pred).query(gt)[0] <= tau
            return gt, fr[2][idx], hit, day
    raise SystemExit(f"{model}/{frame}: no labelled raw frame matched "
                     "(unlabelled day, or a subset-3 plant?)")


def projection(gt, lab, flip=False):
    """2D view: vertical = stem's principal axis, horizontal = widest direction orthogonal to it."""
    c = gt.mean(0)
    stem = gt[lab == por.STEM_ID]
    up = np.linalg.svd(stem - stem.mean(0), full_matrices=False)[2][0]
    if (gt[lab > 0] @ up).mean() < (stem @ up).mean():   # leaves sit above the stem's midpoint
        up = -up
    if flip:
        up = -up
    rest = (gt - c) - np.outer((gt - c) @ up, up)
    side = np.linalg.svd(rest, full_matrices=False)[2][0]
    return lambda x: np.c_[(x - c) @ side, (x - c) @ up]


def fig_frame(models, frame, out_dir, flip, scale):
    data = {m: frame_data(m, frame, scale) for m in models}
    gt0, lab0, _, day = data[models[0]]
    proj = projection(gt0, lab0, flip)

    organ_rec = {}
    for m, (gt, lab, hit, _) in data.items():
        r = np.full(len(gt), np.nan)
        for o in np.unique(lab[lab >= 0]):
            k = lab == o
            r[k] = hit[k].mean()
        organ_rec[m] = r
    allr = np.concatenate([r[~np.isnan(r)] for r in organ_rec.values()])
    vmin, vmax = np.percentile(allr, [2, 98])

    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n + 1.0, 7.0), squeeze=False,
                             constrained_layout=True)
    for j, m in enumerate(models):
        gt, lab, hit, _ = data[m]
        xy = proj(gt)
        r = organ_rec[m]
        un = np.isnan(r)

        ax = axes[0, j]
        ax.scatter(*xy[un].T, s=0.6, c="0.85", lw=0, rasterized=True)
        sc = ax.scatter(*xy[~un].T, s=0.6, c=r[~un], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                        lw=0, rasterized=True)
        ax.set_title(f"{name(m)}\noverall recall {hit.mean():.3f}", fontsize=9)

        ax = axes[1, j]
        ax.scatter(*xy[hit].T, s=0.6, c="0.82", lw=0, rasterized=True)
        ax.scatter(*xy[~hit].T, s=0.6, c="crimson", lw=0, rasterized=True)
        ax.set_title(f"missed GT points: {(~hit).mean():.1%}", fontsize=9)

    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.7, label="recall of the organ at τ")
    plant = frame.rsplit("_", 1)[0]
    fig.suptitle(f"{plant}, day {day} — top: organ recall; bottom: missed GT points (red)",
                 fontsize=10)
    save(fig, out_dir, f"per_organ_{frame}")


def fig_distribution(csv, models, pair, out_dir):
    d = pd.read_csv(csv)
    d = d[(d.source == "instance") & (d.organ > 0) & (d.n_gt >= por.MIN_POINTS)
          & d.model.isin(models)]
    if d.empty:
        print(f"no instance-labelled leaf rows for {models} in {csv}; skipping figure 2")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)

    for m in models:
        r = np.sort(d[d.model == m].recall.values)
        if len(r):
            a.step(r, np.arange(1, len(r) + 1) / len(r), where="post", label=name(m))
    a.set_xlabel("per-leaf recall at τ")
    a.set_ylabel("cumulative fraction of leaves")
    a.set_title(f"lower-right is better (n={d.groupby('model').size().max()} leaf-frames)",
                fontsize=9)
    a.legend(fontsize=8)

    p = d.pivot_table(index=["plant", "day", "organ"], columns="model", values="recall")
    x, y = pair
    if x in p.columns and y in p.columns:
        q = p[[x, y]].dropna()
        for plant, g in q.groupby(level="plant"):
            b.scatter(g[x], g[y], s=12, alpha=0.7, label=plant)
        lo, hi = q.values.min() - 0.02, q.values.max() + 0.02
        b.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        b.set_xlim(lo, hi)
        b.set_ylim(lo, hi)
        b.set_aspect("equal")
        b.set_xlabel(f"{name(x)}: leaf recall")
        b.set_ylabel(f"{name(y)}: leaf recall")
        b.set_title(f"{(q[y] > q[x]).mean():.0%} of leaves above the diagonal (n={len(q)})",
                    fontsize=9)
        b.legend(fontsize=8)
    else:
        b.set_visible(False)
        print(f"pair {pair} not both in {csv}; paired panel skipped")
    save(fig, out_dir, "per_leaf_recall_distribution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", help="e.g. plant_03_0040 (a *_gt.ply stem in cdf_stats/<model>/)")
    ap.add_argument("--models", nargs="+", default=["baseline", "w005", "use_refine_v4"])
    ap.add_argument("--dist_models", nargs="+",
                    default=["baseline", "w000", "w005", "w010", "use_refine_v4"])
    ap.add_argument("--pair", nargs=2, default=["baseline", "w005"])
    ap.add_argument("--csv", default="per_organ_recall.csv")
    ap.add_argument("--out_dir", default="figures_per_organ")
    ap.add_argument("--flip", action="store_true", help="flip the plant if it renders upside down")
    args = ap.parse_args()

    with open(os.path.join(por.DATA, "scale.json")) as f:
        scale = json.load(f)["global_scale"]
    if args.frame is None:
        cands = [os.path.basename(p)[:-len("_gt.ply")] for p in
                 glob.glob(os.path.join(por.CDF, args.models[0], "plant_03_*_gt.ply"))]
        args.frame = min(cands, key=lambda f: abs(int(f.rsplit("_", 1)[1]) - 40))
        print(f"using frame {args.frame}")

    fig_frame(args.models, args.frame, args.out_dir, args.flip, scale)
    if os.path.exists(args.csv):
        fig_distribution(args.csv, args.dist_models, args.pair, args.out_dir)
    else:
        print(f"{args.csv} not found; run per_organ_recall.py first for figure 2")


if __name__ == "__main__":
    main()