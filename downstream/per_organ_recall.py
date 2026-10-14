#!/usr/bin/env python
"""
Per-organ recall on Pepper-4D test frames, computed from the PLYs that
eval_ckpt_claude_v2.py already wrote to cdf_stats/<model>/ (complete-cloud basis).
No GPU and no inference needed.

For every <plant>_<NNNN>_gt.ply the script:
  1. finds the raw frame it came from (NNNN is taken as the capture day),
  2. undoes the dataset normalisation, (pts - base) / scale, and matches each GT
     point to its row in the raw .npy -- exact to float precision and verified,
     so no seeds or dataset code are involved,
  3. labels GT points from labels/instance (subset 1) or labels/temporal (subset 3);
     label -2 is semantic stem without an instance ID, so it is merged into the stem,
  4. reports recall at tau = mean GT nearest-neighbour spacing, grouped by organ.

Usage:
  python per_organ_recall.py --check subset_1/plant_03 0          # label sanity check
  python per_organ_recall.py --models baseline w000 w005 w010 use_refine_v4
"""
import argparse
import glob
import io
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = "/project/cvanddl-cea/sgaihre/Temporal_growth_prediction/Pepper_dataset"
CDF = "/project/cvanddl-cea/sgaihre/cdf_stats"
MATCH_TOL = 1e-5     # raw units; label files are rounded to 6 decimals
UNLABELLED = -1
STEM_ID = 0
MIN_POINTS = 20      # leaves with fewer GT points in the 16,384 subsample are left out of leaf stats
GT_RE = re.compile(r"(plant_\d+)_(\d+)_gt\.ply$")


# ---------------------------------------------------------------- file helpers
def read_ply_xyz(path):
    """Minimal PLY reader (ascii or binary) returning vertex x, y, z as float64."""
    with open(path, "rb") as f:
        fmt, n, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                in_vertex = line.split()[1] == "vertex"
                if in_vertex:
                    n = int(line.split()[2])
            elif line.startswith("property") and in_vertex:
                props.append((line.split()[2], line.split()[1]))
            elif line == "end_header":
                break
        names = [p[0] for p in props]
        if fmt == "ascii":
            arr = np.loadtxt(io.StringIO(f.read().decode()), max_rows=n)
            return arr[:, [names.index(c) for c in "xyz"]].astype(np.float64)
        tmap = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                "short": "i2", "ushort": "u2", "int": "i4", "int32": "i4", "uint": "u4"}
        end = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(nm, end + tmap[t]) for nm, t in props])
        v = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def plant_dir(plant):
    return glob.glob(os.path.join(DATA, "subset_*", plant))[0]


def find_file(folder, day, suffix=".txt"):
    hits = glob.glob(os.path.join(folder, f"*_Day_{day:02d}{suffix}"))
    return hits[0] if hits else None


def match_labels(tree, n_raw, path):
    lab = np.loadtxt(path)
    d, idx = tree.query(lab[:, :3])
    ok = d < MATCH_TOL
    out = np.full(n_raw, UNLABELLED, dtype=np.int64)
    out[idx[ok]] = np.rint(lab[ok, -1]).astype(np.int64)   # label is the LAST column (subset-3 files are x y z r g b label)
    return out


_raw_cache = {}


def raw_frame(pdir, day):
    """(raw xyz, KD-tree, per-row labels or None, label source) for one frame; cached."""
    key = (pdir, day)
    if key not in _raw_cache:
        f = find_file(os.path.join(pdir, "frames"), day, ".npy")
        if f is None:
            _raw_cache[key] = None
        else:
            raw = np.load(f)[:, :3].astype(np.float64)
            tree = cKDTree(raw)
            labels, kind = None, None
            for k in ("instance", "temporal"):
                lf = find_file(os.path.join(pdir, "labels", k), day)
                if lf:
                    labels, kind = match_labels(tree, len(raw), lf), k
                    labels[labels == -2] = STEM_ID
                    break
            _raw_cache[key] = (raw, tree, labels, kind)
    return _raw_cache[key]


def align_to_frame(gt, raw, tree, scale):
    """Find s, t with gt * s + t equal to a subset of raw rows; return row indices or None."""
    for s in (scale, 1.0):            # 1.0 covers PLYs saved in raw units
        g = gt * s
        t = raw.mean(0) - g.mean(0)
        for _ in range(30):
            d, idx = tree.query(g + t)
            if np.median(d) < MATCH_TOL:
                break
            t = t + np.median(raw[idx] - (g + t), axis=0)
        d, idx = tree.query(g + t)
        if d.max() < MATCH_TOL:
            return idx
    return None


# ---------------------------------------------------------------- modes
def check(plant_rel, day):
    pdir = os.path.join(DATA, plant_rel)
    raw = np.load(find_file(os.path.join(pdir, "frames"), day, ".npy"))[:, :3]
    tree = cKDTree(raw)
    labs = {}
    for k in ("instance", "temporal"):
        lf = find_file(os.path.join(pdir, "labels", k), day)
        if lf:
            labs[k] = match_labels(tree, len(raw), lf)
    sem = glob.glob(os.path.join(pdir, "labels", "semantic", f"*_Day_{day:02d}_semantic.txt"))
    if sem:
        labs["semantic"] = match_labels(tree, len(raw), sem[0])
    print(f"raw points {len(raw)}; label sources: {list(labs)}")
    for k in ("instance", "temporal"):
        if k in labs and "semantic" in labs:
            print(f"\n{k} vs semantic (semantic 0 = stem, 1 = leaf):")
            print(pd.crosstab(pd.Series(labs[k], name=k), pd.Series(labs["semantic"], name="semantic")))
    if "instance" in labs and "temporal" in labs:
        same = np.mean(labs["instance"] == labs["temporal"])
        print(f"\ninstance and temporal IDs identical on {same:.1%} of points")


def run(models, out_csv):
    with open(os.path.join(DATA, "scale.json")) as f:
        scale = json.load(f)["global_scale"]
    rows, overall, failed = [], [], []
    for model in models:
        gts = sorted(glob.glob(os.path.join(CDF, model, "*_gt.ply")))
        print(f"{model}: {len(gts)} frames")
        for gt_path in gts:
            plant, num = GT_RE.search(os.path.basename(gt_path)).groups()
            num = int(num)
            gt = read_ply_xyz(gt_path)
            pred = read_ply_xyz(gt_path.replace("_gt.ply", "_rec.ply"))
            pdir = plant_dir(plant)
            found = None
            for day in dict.fromkeys((num, 2 * num)):
                fr = raw_frame(pdir, day)
                if fr is not None:
                    idx = align_to_frame(gt, fr[0], fr[1], scale)
                    if idx is not None:
                        found = (day, fr, idx)
                        break
            if found is None:
                failed.append(f"{model}/{os.path.basename(gt_path)}")
                continue
            day, (_, _, labels, kind), idx = found
            tau = cKDTree(gt).query(gt, k=2)[0][:, 1].mean()
            hit = cKDTree(pred).query(gt)[0] <= tau
            overall.append(dict(model=model, plant=plant, day=day, recall=hit.mean(),
                                labelled=labels is not None))
            if labels is None:
                continue
            lab = labels[idx]
            for o in np.unique(lab):
                m = lab == o
                rows.append(dict(model=model, plant=plant, day=day, source=kind, organ=int(o),
                                 n_gt=int(m.sum()), hits=int(hit[m].sum()),
                                 recall=float(hit[m].mean())))
    if failed:
        print(f"\nCould not match to a raw frame (skipped): {failed}")
    ov = pd.DataFrame(overall)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    ov.to_csv(out_csv.replace(".csv", "_frames.csv"), index=False)
    summarise(ov, df)


def summarise(ov, df):
    pooled = lambda g: g.hits.sum() / max(g.n_gt.sum(), 1)
    leaves = df[(df.organ > 0) & (df.n_gt >= MIN_POINTS)].copy()
    leaves["size_bin"] = pd.qcut(leaves.n_gt, 3, labels=["small", "medium", "large"])
    for model, g in df.groupby("model"):
        o = ov[ov.model == model]
        print(f"\n=== {model}")
        print(f"  overall recall, all {len(o)} frames (should match Table 1): {o.recall.mean():.3f}")
        print(f"  labelled frames used: {int(o.labelled.sum())}")
        for src, gs in g.groupby("source"):
            lg = leaves[(leaves.model == model) & (leaves.source == src)]
            print(f"  [{src}] stem recall {pooled(gs[gs.organ == STEM_ID]):.3f} | "
                  f"leaves n={len(lg)}: median recall {lg.recall.median():.3f}, "
                  f"share with recall >= 0.5: {(lg.recall >= 0.5).mean():.3f}")
        lg = leaves[leaves.model == model]
        bins = ", ".join(f"{b} {pooled(bg):.3f}" for b, bg in lg.groupby("size_bin", observed=True))
        print(f"  leaf recall by size (all sources): {bins}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs=2, metavar=("PLANT_DIR", "DAY"))
    ap.add_argument("--models", nargs="+", default=["baseline", "w005"])
    ap.add_argument("--out", default="per_organ_recall.csv")
    args = ap.parse_args()
    if args.check:
        check(args.check[0], int(args.check[1]))
    else:
        run(args.models, args.out)


if __name__ == "__main__":
    main()