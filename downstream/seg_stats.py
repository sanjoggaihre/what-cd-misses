#!/usr/bin/env python
"""
Paired statistics for the downstream segmentation test.

Reads the per-frame files written by seg_downstream.py (seg_s*_frames.csv), and for every
method and metric reports the paired difference from a reference model (default: the
Point-MAE baseline):

  mean_diff     mean over frames of (method - reference), each frame first averaged over
                segmenter seeds
  ci_lo, ci_hi  95% bootstrap CI of that mean; frames are resampled WITHIN each plant
                (stratified), so every resample keeps the plant mix of the test set
  per_seed      mean difference computed separately for each segmenter seed
  wins          share of (seed, frame) pairs where the method beats the reference

Caveat to state in the paper: frames of the same plant are correlated and there are only
2-4 test plants, so these CIs describe variation across frames of these plants, not
across plants in general.

Usage:
  python seg_stats.py                              # vs baseline
  python seg_stats.py --ref w005                   # e.g. ours vs repulsion / decoder
  python seg_stats.py --files "seg_s*_frames.csv" --out seg_stats.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd

METRICS = ["inferred_mIoU", "inferred_stem", "on_gt_mIoU", "gated_mIoU", "gated_stem",
           "gated_leaf", "cov_all"]


def stratified_boot_ci(d, plants, n_boot, rng):
    groups = [np.flatnonzero(plants == p) for p in np.unique(plants)]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        means[b] = d[idx].mean()
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="seg_s*_frames.csv")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", default="seg_stats.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.files))
    f = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "plant" not in f.columns:
        f["plant"] = f.frame.str.rsplit("_", n=1).str[0]
    metrics = [m for m in METRICS if m in f.columns]
    seeds = sorted(int(s) for s in f.seed.unique())
    print(f"{len(paths)} files, seeds {seeds}, {f.frame.nunique()} frames, "
          f"plants {sorted(f.plant.unique())}, reference = {args.ref}")
    missing = [m for m in METRICS if m not in f.columns]
    if missing:
        print(f"(metrics not in these files, rerun seg_downstream.py to get them: {missing})")

    rng = np.random.default_rng(0)
    rows = []
    for metric in metrics:
        p = f.pivot_table(index=["seed", "frame", "plant"], columns="model", values=metric)
        if args.ref not in p.columns:
            raise SystemExit(f"reference {args.ref} not found; models: {list(p.columns)}")
        for m in p.columns:
            if m == args.ref:
                continue
            diff = (p[m] - p[args.ref]).dropna()
            per_frame = diff.groupby(level=["frame", "plant"]).mean()
            plants = per_frame.index.get_level_values("plant").to_numpy()
            d = per_frame.to_numpy()
            lo, hi = stratified_boot_ci(d, plants, args.n_boot, rng)
            per_seed = diff.groupby(level="seed").mean()
            rows.append(dict(metric=metric, model=m, ref=args.ref,
                             ref_mean=p[args.ref].groupby(level="frame").mean().mean(),
                             model_mean=p[m].groupby(level="frame").mean().mean(),
                             mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                             excludes_zero=bool(lo > 0 or hi < 0),
                             per_seed=" / ".join(f"{v:+.3f}" for v in per_seed),
                             wins=(diff > 0).mean(), n_frames=len(d)))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        for metric, g in out.groupby("metric", sort=False):
            print(f"\n{metric}")
            print(g.drop(columns=["metric", "ref"]).round(3).to_string(index=False))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()