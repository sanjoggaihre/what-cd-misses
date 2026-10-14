#!/usr/bin/env python3
"""
summarize_per_organ.py -- per-organ summary for every model in a
per_organ_recall.py CSV, using the definition that reproduces the paper.

The CSV has one row per (model, plant, day, source, organ) with the number of
GT points (n_gt), how many have a prediction within tau (hits) and their ratio
(recall). The paper's summary, per label source (instance = plants 03/11,
temporal = plants 28/29), is:
    stem recall                  hits / n_gt pooled over stem rows
    median per-leaf recall       median over leaf observations
    share of leaves >= 0.5       fraction of leaf observations covered at least half
    leaf observations            205 (instance) and 108 (temporal)

Which organ id marks the stem, whether stem recall is pooled or averaged per
frame, whether tiny leaves were dropped, and whether the cut-off is >= or >
were not recorded. This script tries each combination, keeps the one that
reproduces the published baseline and w005 numbers, reports how closely it
does, and only then summarises the other models with it.

Usage:
  python summarize_per_organ.py --csv per_organ_all.csv
Requires numpy and pandas.
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

# Published values (paper, per-organ table): stem, leaf median, share (%), n leaves
PUBLISHED = {
    ("baseline", "instance"): (0.527, 0.519, 55.6, 205),
    ("baseline", "temporal"): (0.634, 0.505, 52.8, 108),
    ("w005", "instance"): (0.569, 0.570, 80.5, 205),
    ("w005", "temporal"): (0.666, 0.557, 72.2, 108),
    ("w000", "instance"): (0.524, 0.517, 56.1, 205),
    ("w000", "temporal"): (0.632, 0.500, 50.0, 108),
    ("w010", "instance"): (0.559, 0.558, 77.1, 205),
    ("w010", "temporal"): (0.660, 0.549, 71.3, 108),
}
NAMES = {"baseline": "Point-MAE (CD)", "w000": "w_sp=0 control", "w005": "Ours (w_sp=0.005)",
         "w010": "Ours (w_sp=0.010)", "dcd_a50": "DCD a=50, l=1", "dcd_a50_l05": "DCD a=50, l=0.5",
         "dcd_a200": "DCD a=200, l=1", "dcd_a200_l05": "DCD a=200, l=0.5",
         "rep_h007_w1": "Repulsion h=.007, w=1", "rep_h007_w05": "Repulsion h=.007, w=0.5"}


def summarise(df, stem_id, pooled, min_n, strict, exclude):
    """(stem, leaf_median, share %, n_leaves) for one model/source slice."""
    stem = df[df.organ == stem_id]
    if len(stem) == 0 or stem.n_gt.sum() == 0:
        s = np.nan
    elif pooled:
        s = stem.hits.sum() / stem.n_gt.sum()
    else:
        s = stem.groupby(["plant", "day"]).apply(
            lambda g: g.hits.sum() / g.n_gt.sum()).mean()
    leaves = df[~df.organ.isin([stem_id] + exclude) & (df.n_gt >= min_n)]
    r = leaves.recall.to_numpy(float)
    if len(r) == 0:
        return s, np.nan, np.nan, 0
    share = 100 * ((r > 0.5) if strict else (r >= 0.5)).mean()
    return s, float(np.median(r)), float(share), len(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tol", type=float, default=0.0025,
                    help="max abs error on recalls (and tol*100 points on the share)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"model", "plant", "day", "source", "organ", "n_gt", "hits", "recall"}
    if not need <= set(df.columns):
        sys.exit(f"CSV columns {list(df.columns)} lack {need - set(df.columns)}")
    df["organ"] = pd.to_numeric(df.organ, errors="coerce")
    sources = sorted(df.source.unique())
    models = list(dict.fromkeys(df.model))
    print(f"models: {models}\nsources: {sources}")
    for src in sources:
        top = df[(df.source == src) & (df.model == models[0])].organ.value_counts().head(6)
        print(f"  {src}: most frequent organ ids {dict(top)}")

    refs = [(m, s) for (m, s) in PUBLISHED if m in models and s in sources]
    if not refs:
        sys.exit("None of baseline/w000/w005/w010 is in the CSV, so the definition "
                 "cannot be checked against the paper.")

    organ_ids = sorted(df.organ.dropna().unique())
    stem_cands = [i for i in (-1, 0) if i in organ_ids] or organ_ids[:2]
    grid = itertools.product(stem_cands, (True, False), (0, 1, 5, 10, 20, 30, 50, 100),
                             (False, True), ([], [-1], [0]))
    best = None
    for stem_id, pooled, min_n, strict, excl in grid:
        if stem_id in excl:
            continue
        err, count_err = 0.0, 0
        for m, s in refs:
            pub = PUBLISHED[(m, s)]
            got = summarise(df[(df.model == m) & (df.source == s)], stem_id, pooled,
                            min_n, strict, excl)
            if np.isnan(got[0]) or np.isnan(got[1]):
                err = np.inf
                break
            err = max(err, abs(got[0] - pub[0]), abs(got[1] - pub[1]),
                      abs(got[2] - pub[2]) / 100)
            count_err += abs(got[3] - pub[3])
        key = (count_err, err)
        if best is None or key < best[0]:
            best = (key, dict(stem_id=stem_id, pooled=pooled, min_n=min_n, strict=strict,
                              exclude=excl))

    (count_err, err), cfg = best
    print(f"\nBest-matching definition: stem organ id {cfg['stem_id']}, stem recall "
          f"{'pooled over points' if cfg['pooled'] else 'averaged over frames'}, leaves with "
          f"at least {cfg['min_n']} GT points, share uses "
          f"{'> 0.5' if cfg['strict'] else '>= 0.5'}, extra excluded ids {cfg['exclude']}")
    print(f"Against the published numbers: max error {err:.4f}, leaf-count mismatch "
          f"{count_err} (checked on {', '.join(f'{m}/{s}' for m, s in refs)})")
    ok = err <= args.tol and count_err == 0
    if not ok:
        print("\n[warn] No definition reproduces the published numbers within tolerance. "
              "The CSV may come from different evaluation settings; do not mix these "
              "numbers with the published ones.", file=sys.stderr)

    print(f"\n{'model':26s} {'source':9s} {'stem':>6} {'leaf med':>9} {'>=half %':>9} "
          f"{'n leaves':>9}  published")
    rows = []
    for m in models:
        for s in sources:
            got = summarise(df[(df.model == m) & (df.source == s)], cfg["stem_id"],
                            cfg["pooled"], cfg["min_n"], cfg["strict"], cfg["exclude"])
            pub = PUBLISHED.get((m, s))
            pub_txt = f"{pub[0]:.3f} {pub[1]:.3f} {pub[2]:.1f}" if pub else ""
            print(f"{NAMES.get(m, m):26s} {s:9s} {got[0]:6.3f} {got[1]:9.3f} {got[2]:9.1f} "
                  f"{got[3]:9d}  {pub_txt}")
            rows.append((m, s) + got)

    # Paired comparison of every model against ours, leaf by leaf.
    if "w005" in models:
        print("\nPaired per-leaf comparison against Ours (w_sp=0.005): mean recall "
              "difference (model - ours) and share of leaves where ours is higher")
        key = ["plant", "day", "source", "organ"]
        leaf = lambda m: df[(df.model == m) & ~df.organ.isin([cfg["stem_id"]] + cfg["exclude"])
                            & (df.n_gt >= cfg["min_n"])].set_index(key).recall
        ours = leaf("w005")
        for m in models:
            if m == "w005":
                continue
            j = pd.concat([leaf(m).rename("m"), ours.rename("o")], axis=1).dropna()
            print(f"  {NAMES.get(m, m):26s} diff {np.mean(j.m - j.o):+.3f}   ours higher on "
                  f"{100 * np.mean(j.o > j.m):.1f}% of {len(j)} leaves")

    pd.DataFrame(rows, columns=["model", "source", "stem_recall", "leaf_median",
                                "share_ge_half", "n_leaves"]).to_csv(
        args.csv.replace(".csv", "_summary.csv"), index=False)
    print(f"\nSummary written to {args.csv.replace('.csv', '_summary.csv')}")


if __name__ == "__main__":
    main()