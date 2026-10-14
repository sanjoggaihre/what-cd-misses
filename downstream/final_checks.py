#!/usr/bin/env python3
"""
final_checks.py -- recompute the paper's quoted numbers from your saved outputs.

1) cdf: error-distribution numbers (Sec. 3.2, Sec. 5.1, Sec. 5.5, Fig. 1 caption)
   Reads <root>/<model>/<frame>_gt.ply and <frame>_rec.ply, normalises each
   frame's distances by that frame's tau (mean GT nearest-neighbour spacing),
   pools them over frames and prints, per direction:
     n            pooled distances (paper: 376,832 = 23 x 16,384 per direction)
     =0           share at exactly zero (copied points; Fig. 1 caption ~0.40 / ~0.35)
     <=0.1 <=0.2  near-zero region (paper: 0.412 vs 0.401 within 0.1 spacings)
     <=1          precision / recall at tau (Table 1)
     <=2          paper: precision 0.896 -> 0.775, recall 0.774 -> 0.868
     median p95   paper: Point-MAE pred->GT median 0.263, GT->pred median 0.996;
                  assembled-CD control GT->pred median 1.003, p95 3.216
     >10          share beyond 10 tau (paper: 2.863% for DCD alpha=200, lambda=1)

   python final_checks.py cdf --root /project/cvanddl-cea/sgaihre/cdf_stats \
       --models baseline w005 w000

2) seg: unrounded downstream means and differences (Table 4 and the text)
   Pools seg_s0/s1/s2 per-frame CSVs, averages every numeric column per model
   and prints the means and each model's difference from --ref to 4 decimals.
   Paper text: 0.151 / 0.252 vs Point-MAE, 0.134 stem and 0.043 gated vs
   DCD lambda=0.5, 0.006 gated vs DCD lambda=1, 0.018 gated / 0.017 coverage
   vs repulsion w=1.

   python final_checks.py seg --csv "seg_s*_frames.csv" --ref w005

Requires numpy and scipy (cdf) or pandas (seg).
"""

import argparse
import glob
import os
import sys

import numpy as np

_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
              "uint": "u4", "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1",
              "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4",
              "float32": "f4", "float64": "f8"}


def read_ply_xyz(path):
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, n, props, in_vertex = None, None, [], False
        while True:
            tok = fh.readline().decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                props.append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        names = [nm for nm, _ in props]
        if fmt == "ascii":
            data = np.loadtxt(fh, max_rows=n, ndmin=2)
            return data[:, [names.index(c) for c in "xyz"]].astype(np.float64)
        dt = np.dtype([(nm, ("<" if fmt == "binary_little_endian" else ">") + t)
                       for nm, t in props])
        arr = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def summarise(d):
    q = np.percentile(d, [50, 95])
    return (f"n={len(d):>8,d}  =0 {np.mean(d == 0):.3f}  <=0.1 {np.mean(d <= 0.1):.3f}  "
            f"<=0.2 {np.mean(d <= 0.2):.3f}  <=1 {np.mean(d <= 1):.3f}  <=2 {np.mean(d <= 2):.3f}  "
            f"median {q[0]:.3f}  p95 {q[1]:.3f}  >10 {100 * np.mean(d > 10):.3f}%")


def run_cdf(args):
    from scipy.spatial import cKDTree
    for model in args.models:
        gts = sorted(glob.glob(os.path.join(args.root, model, "*_gt.ply")))
        if not gts:
            print(f"{model}: no *_gt.ply in {os.path.join(args.root, model)}")
            continue
        pg, gp, n_rec = [], [], set()
        for g in gts:
            r = g[:-len("_gt.ply")] + "_rec.ply"
            if not os.path.exists(r):
                print(f"  missing {os.path.basename(r)}; skipped")
                continue
            G, R = read_ply_xyz(g), read_ply_xyz(r)
            tg = cKDTree(G)
            tau = tg.query(G, k=2)[0][:, 1].mean()
            pg.append(tg.query(R)[0] / tau)
            gp.append(cKDTree(R).query(G)[0] / tau)
            n_rec.add(len(R))
        print(f"\n{model}: {len(pg)} frames, rec points per frame {sorted(n_rec)}")
        print("  pred->GT (a) " + summarise(np.concatenate(pg)))
        print("  GT->pred (b) " + summarise(np.concatenate(gp)))


def run_seg(args):
    import pandas as pd
    files = sorted(glob.glob(args.csv))
    if not files:
        sys.exit("No files matched --csv (quote the pattern).")
    df = pd.concat([pd.read_csv(f).assign(_file=os.path.basename(f)) for f in files])
    if "model" not in df.columns:
        sys.exit(f"No 'model' column. Columns: {list(df.columns)}")
    num = [c for c in df.select_dtypes("number").columns if c not in ("seed", "day", "frame")]
    print(f"{len(files)} files: {[os.path.basename(f) for f in files]}")
    print("rows per model (20 frames x 3 seeds = 60 expected):")
    print(df.groupby("model").size().to_string())
    means = df.groupby("model")[num].mean()
    pd.set_option("display.width", 200)
    print("\nmeans (4 dp):")
    print(means.round(4).to_string())
    if args.ref in means.index:
        print(f"\n{args.ref} minus each model (4 dp):")
        print((means.loc[args.ref] - means).drop(index=args.ref).round(4).to_string())
    else:
        print(f"\n--ref {args.ref} not found; models are {list(means.index)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cdf")
    a.add_argument("--root", required=True)
    a.add_argument("--models", nargs="+", required=True)
    b = sub.add_parser("seg")
    b.add_argument("--csv", required=True)
    b.add_argument("--ref", default="w005")
    args = ap.parse_args()
    run_cdf(args) if args.cmd == "cdf" else run_seg(args)


if __name__ == "__main__":
    main()