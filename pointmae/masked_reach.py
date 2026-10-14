#!/usr/bin/env python3
"""
masked_reach.py -- how much of the complete cloud lies outside every masked patch?

Our coverage term and DCD both compare the masked predictions with the COMPLETE
input cloud. Points that belong to no masked patch are ones no masked prediction is
meant to reconstruct, so they set a floor under both terms. This script replicates
Point-MAE's grouping (FPS from index 0, 512 centres, 64 nearest neighbours each,
int(0.6 * 512) = 307 patches masked at random) on real frames and reports:

  outside      share of cloud points that belong to no masked patch
  floor_tau    the coverage term (mean of the largest 10% of ground-truth-to-
               prediction distances) for a PERFECT reconstruction of every
               masked patch, in units of tau (mean NN spacing of the frame)
  tail_outside share of that perfect-reconstruction top-10% tail lying outside
               every masked patch (100% whenever outside >= 10%)

Usage (any model folder works; only the *_gt.ply files are read):
  python masked_reach.py --gt_glob "cdf_stats/baseline/*_gt.ply"
Requires numpy and scipy.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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


def fps(X, n):
    idx = np.zeros(n, int)                      # starts from point 0, as pointnet2 does
    d = np.full(len(X), np.inf)
    for i in range(1, n):
        d = np.minimum(d, ((X - X[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(d.argmax())
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_glob", required=True)
    ap.add_argument("--groups", type=int, default=512)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--n_masks", type=int, default=20, help="random masks per frame")
    ap.add_argument("--q", type=float, default=0.1, help="tail fraction of the coverage term")
    args = ap.parse_args()

    files = sorted(glob.glob(args.gt_glob))
    if not files:
        raise SystemExit("No files matched --gt_glob (quote the pattern).")
    n_mask = int(args.mask_ratio * args.groups)
    print(f"{len(files)} frames; {n_mask} of {args.groups} patches masked -> "
          f"{n_mask * args.k} predicted points per sample\n")
    rng = np.random.default_rng(0)
    rows = []
    for f in files:
        X = read_ply_xyz(f)
        tree = cKDTree(X)
        tau = tree.query(X, k=2)[0][:, 1].mean()
        _, nb = tree.query(X[fps(X, args.groups)], k=args.k)       # (G, k)
        out, floor, tail_out = [], [], []
        for _ in range(args.n_masks):
            masked = np.zeros(args.groups, bool)
            masked[rng.choice(args.groups, n_mask, replace=False)] = True
            inside = np.zeros(len(X), bool)
            inside[nb[masked].ravel()] = True
            d = cKDTree(X[inside]).query(X)[0]                      # perfect masked reconstruction
            top = np.argsort(d)[-max(1, int(args.q * len(X))):]
            out.append(1 - inside.mean())
            floor.append(d[top].mean() / tau)
            tail_out.append((~inside[top]).mean())
        rows.append((np.mean(out), np.mean(floor), np.mean(tail_out)))
        print(f"{Path(f).name:28s} outside {100 * rows[-1][0]:5.1f}%   floor {rows[-1][1]:5.2f} tau"
              f"   tail outside {100 * rows[-1][2]:5.1f}%")
    r = np.array(rows)
    print(f"\nMean over frames: outside {100 * r[:, 0].mean():.1f}% "
          f"(range {100 * r[:, 0].min():.1f}-{100 * r[:, 0].max():.1f}%), "
          f"coverage-term floor {r[:, 1].mean():.2f} tau, tail outside {100 * r[:, 2].mean():.1f}%")


if __name__ == "__main__":
    main()