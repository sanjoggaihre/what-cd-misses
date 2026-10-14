#!/usr/bin/env python
"""
oracle.py -- the reconstruction floor at this cardinality.

Draws two INDEPENDENT 16,384-point subsamples from the same raw scan and
scores one against the other with the identical metric suite used for the
models. A perfect reconstruction cannot do better than this: the residual is
pure sampling error, not model error.

Without this reference, every absolute number in the results is
uninterpretable -- CD-L1 of 1.7e-3 could be near-perfect or twice the
achievable error, and there is no way to tell.

Metric functions are imported from eval_ckpt_claude_v2 rather than
reimplemented, so the oracle and the models are measured by the same code.

Usage:
    python oracle.py --config cfgs/pretrain_pepper_baseline.yaml \\
                     --out_dir /project/.../cdf_stats/oracle
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from utils.config import cfg_from_yaml_file          # noqa: E402
from datasets import build_dataset_from_cfg          # noqa: E402
from eval_ckpt_claude_v2 import (                    # noqa: E402
    chamfer_l1, chamfer_l2, spacing_ratio, spacing_cv, fscore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=20,
                    help="frames to sample; matches the model evaluations")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = cfg_from_yaml_file(args.config)
    dcfg = getattr(cfg.dataset, args.split)
    ds = build_dataset_from_cfg(dcfg._base_, dcfg.others)
    print(f":: {args.split}: {len(ds)} frames\n")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    step = max(1, len(ds) // args.n)
    for i in range(0, len(ds), step):
        # Two draws with different seeds. The dataset subsamples with
        # np.random.choice at __getitem__ time, so reseeding between calls
        # yields two independent 16,384-point views of the same scan.
        np.random.seed(1000 + i)                 # same seed the evals use
        d0 = ds[i][2]
        a = (d0["points"] if isinstance(d0, dict) else d0).numpy()

        np.random.seed(50000 + i)                # independent second draw
        d1 = ds[i][2]
        b = (d1["points"] if isinstance(d1, dict) else d1).numpy()

        if np.array_equal(a, b):
            raise SystemExit(
                "the two draws are identical -- the dataset is caching the "
                "subsample, not just the raw file. Oracle is meaningless "
                "until that is resolved.")

        f1, prec, rec = fscore(a, b)

        # Normalised distance distributions, same convention as the evals.
        d_pg, _ = cKDTree(b).query(a, workers=-1)
        d_gp, _ = cKDTree(a).query(b, workers=-1)
        dg, _ = cKDTree(b).query(b, k=2, workers=-1)
        s = dg[:, 1].mean()

        rows.append({
            "id": str(ds[i][1]), "n": len(a),
            "cd_l1": chamfer_l1(b, a), "cd_l2": chamfer_l2(b, a),
            "spacing": spacing_ratio(a, b),
            "cv": spacing_cv(a), "cv_gt": spacing_cv(b),
            "f1": f1, "prec": prec, "recall": rec,
            "pg_med": float(np.median(d_pg / s)),
            "pg_p90": float(np.percentile(d_pg / s, 90)),
            "pg_p95": float(np.percentile(d_pg / s, 95)),
            "gp_med": float(np.median(d_gp / s)),
            "gp_p90": float(np.percentile(d_gp / s, 90)),
            "gp_p95": float(np.percentile(d_gp / s, 95)),
        })

        safe = str(ds[i][1]).replace("/", "_").replace(".npy", "")
        np.savez_compressed(os.path.join(args.out_dir, f"{safe}_dists.npz"),
                            pg=(d_pg / s).astype(np.float32),
                            gp=(d_gp / s).astype(np.float32))

    print(f"{'Frame ID':<24}{'CD-L1':>12}{'Spacing':>9}{'CV':>7}"
          f"{'F1':>7}{'Prec':>7}{'Rec':>7}")
    print("-" * 73)
    for r in rows:
        print(f"{r['id'][:23]:<24}{r['cd_l1']:>12.4e}{r['spacing']:>9.3f}"
              f"{r['cv']:>7.3f}{r['f1']:>7.3f}{r['prec']:>7.3f}"
              f"{r['recall']:>7.3f}")
    print("-" * 73)

    print("\n=== ORACLE (floor at 16,384 points) ===")
    for k, label in (("cd_l1", "CD-L1"), ("cd_l2", "CD-L2"),
                     ("spacing", "Spacing"), ("cv", "CV"), ("cv_gt", "CV_gt"),
                     ("f1", "F1"), ("prec", "Prec"), ("recall", "Recall"),
                     ("pg_med", "P->G med"), ("pg_p90", "P->G p90"),
                     ("pg_p95", "P->G p95"),
                     ("gp_med", "G->P med"), ("gp_p90", "G->P p90"),
                     ("gp_p95", "G->P p95")):
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"  {label:<10} {v.mean():.4e}  +/- {v.std():.4e}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()