#!/usr/bin/env python3
"""
eval_ckpt_claude_v2.py -- Evaluate a trained Point-MAE / Point_MAE_Hybrid checkpoint.

    python eval_ckpt_claude_v2.py --config cfgs/pretrain_pepper_hybrid.yaml \
        --ckpt experiments/.../ckpt-best.pth --split test \
        --dense_only --out_dir reconstructions/hybrid

Reports CD-L1, CD-L2 and EMD in scientific notation. Values here land around
1e-5 to 1e-3, so %.5f rounds most of them to 0.00000 and destroys the
comparison -- %.4e keeps four significant figures whatever the magnitude.

Both CD variants are reported because papers differ on which they mean:
  CD-L2  sum of MEAN SQUARED nearest-neighbour distances (PCN convention)
  CD-L1  mean of MEAN euclidean distances (SeedFormer / PoinTr convention)
CD-L2 is ~the square of CD-L1, hence far smaller and easier to misread.

EMD requires equal point counts (it is a bijection), so the prediction is
subsampled to the ground-truth size before it is computed. Exact EMD is
O(n^3); this uses a Sinkhorn approximation, and is skipped above
--emd-max-points because cost grows with n^2 in memory.
"""

import argparse
import os
import sys
import json

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from utils.config import cfg_from_yaml_file          # noqa: E402
from tools import builder                            # noqa: E402
from datasets import build_dataset_from_cfg          # noqa: E402


# ------------------------------------------------------------------ metrics
def chamfer_l2(a, b):
    """Sum of mean SQUARED nearest-neighbour distances (PCN convention)."""
    d1, _ = cKDTree(b).query(a, workers=-1)
    d2, _ = cKDTree(a).query(b, workers=-1)
    return float((d1 ** 2).mean() + (d2 ** 2).mean())


def chamfer_l1(a, b):
    """Mean of the two mean euclidean distances (SeedFormer convention)."""
    d1, _ = cKDTree(b).query(a, workers=-1)
    d2, _ = cKDTree(a).query(b, workers=-1)
    return float(0.5 * (d1.mean() + d2.mean()))

def spacing_cv(pts):
    """Coefficient of variation of nearest-neighbour distance. Captures
    uniformity, which the mean-spacing ratio cannot: dense knots plus gaps
    can average to a correct mean."""
    d, _ = cKDTree(pts).query(pts, k=2, workers=-1)
    nn = d[:, 1]
    return float(nn.std() / nn.mean())

def spacing_ratio(rec, gt):
    """Mean nearest-neighbour spacing of the prediction over that of the GT.

    1.0 is ideal. Below 1 means points sit closer together than the data
    warrants -- clumping. Above 1 means the cloud has spread out, which
    Chamfer distance cannot see either, since neither duplication nor
    thinning costs it anything.
    """
    dp, _ = cKDTree(rec).query(rec, k=2, workers=-1)
    dg, _ = cKDTree(gt).query(gt, k=2, workers=-1)
    return float(dp[:, 1].mean() / dg[:, 1].mean())

def fscore(rec, gt, tau=None):
    """F-score at threshold tau, the standard point-completion metric.

    precision: fraction of predicted points within tau of some GT point
    recall:    fraction of GT points within tau of some predicted point

    Unlike Chamfer, this separates the two directions, so holes (low recall)
    and stray points (low precision) are visible instead of averaged
    together. tau defaults to the GT's own mean NN distance, which makes the
    number scale-free and comparable across datasets.
    """
    dg, _ = cKDTree(gt).query(gt, k=2, workers=-1)
    if tau is None:
        tau = dg[:, 1].mean()

    d_pred, _ = cKDTree(gt).query(rec, workers=-1)
    d_gt, _ = cKDTree(rec).query(gt, workers=-1)

    p = float((d_pred < tau).mean())
    r = float((d_gt < tau).mean())
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f, p, r

def distance_stats(pred, gt):
    """Full distribution of both nearest-neighbour error directions.

    Chamfer reports only the means. Two clouds with equal CD can differ
    sharply in their tails: many small errors versus few large ones.
    Distances are normalised by the GT's own mean NN spacing, so values read
    as multiples of inter-point distance.
    """
    d_pg, _ = cKDTree(gt).query(pred, workers=-1)
    d_gp, _ = cKDTree(pred).query(gt, workers=-1)
    dg, _ = cKDTree(gt).query(gt, k=2, workers=-1)
    s = dg[:, 1].mean()

    out = {}
    for name, d in (("pg", d_pg / s), ("gp", d_gp / s)):
        out[f"{name}_med"] = float(np.median(d))
        out[f"{name}_p90"] = float(np.percentile(d, 90))
        out[f"{name}_p95"] = float(np.percentile(d, 95))
        out[f"{name}_max"] = float(d.max())
    return out

def emd_sinkhorn(a, b, eps=0.01, iters=200, device='cuda'):
    """Entropic-regularised EMD (Sinkhorn), in log space.

    The direct-space version underflows: with eps small relative to the
    cloud's point spacing, exp(-C/(eps*scale)) is zero for all but the
    nearest neighbours in float32, K becomes singular, and the result
    collapses toward a nearest-neighbour cost -- i.e. it reports CD.
    """
    n = min(len(a), len(b))
    rng = np.random.default_rng(0)
    if len(a) > n:
        a = a[rng.choice(len(a), n, replace=False)]
    if len(b) > n:
        b = b[rng.choice(len(b), n, replace=False)]

    A = torch.as_tensor(a, dtype=torch.float32, device=device)
    B = torch.as_tensor(b, dtype=torch.float32, device=device)

    C = torch.cdist(A, B)
    scale = float((A.max(0).values - A.min(0).values).max())
    reg = eps * scale

    log_mu = torch.full((n,), -np.log(n), device=device)
    f = torch.zeros(n, device=device)
    g = torch.zeros(n, device=device)

    for _ in range(iters):
        f = reg * (log_mu - torch.logsumexp((g.unsqueeze(0) - C) / reg, dim=1)) + f
        g = reg * (log_mu - torch.logsumexp((f.unsqueeze(1) - C) / reg, dim=0)) + g

    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / reg)
    return float((P * C).sum())


# ----------------------------------------------------------- reconstruction
def reconstruct(model, pts, dense_only=False, mode="full", n_target=None):
    """
    dense_only  -- return out[0] alone. REQUIRED for Point_MAE_Hybrid, whose
                   out[1] is the coarse 2048-point scaffold, not visible
                   patches. Including it cost 0.044 F in earlier runs.
    mode="full" -- concatenate visible + rebuilt. For Point_MAE this means
                   ~40% of the output is COPIED GROUND TRUTH at mask_ratio
                   0.6, which inflates every metric.
    mode="masked" -- drop points coinciding with visible patches; the honest
                   measure of what the model inferred.
    """
    with torch.no_grad():
        out = model(pts, vis=True)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError("unexpected model output; check forward(vis=True)")

    pred = out[0].reshape(-1, 3).cpu().numpy()
    if dense_only:
        rec = pred
    else:
        vis = out[1].reshape(-1, 3).cpu().numpy()
        if mode == "masked":
            d, _ = cKDTree(vis).query(pred, workers=-1)
            rec = pred[d > 1e-6]
        else:
            rec = np.vstack([vis, pred])

    if n_target is not None and len(rec) > n_target:
        idx = np.random.default_rng(0).choice(len(rec), n_target, replace=False)
        rec = rec[idx]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default="reconstructions")
    ap.add_argument("--dense_only", action="store_true",
                    help="return out[0] only (required for the hybrid)")
    ap.add_argument("--mode", default="full", choices=["full", "masked"])
    ap.add_argument("--match_density", action="store_true",
                    help="subsample prediction to the GT point count")
    ap.add_argument("--no_emd", action="store_true")
    ap.add_argument("--emd-max-points", type=int, default=16384,
                    help="skip EMD above this; the cost matrix is n^2")
    ap.add_argument("--mask_ratio", type=float, default=None,
                    help="override the config mask ratio; 0.0 evaluates "
                         "reconstruction rather than 60%% inpainting")
    ap.add_argument("--n_points", type=int, default=None,
                    help="subsample the output to this many points before "
                         "metrics; use with --dense_only to measure density "
                         "on the complete cloud at matched cardinality")
    args = ap.parse_args()

    if args.split == "test":
        print("!! Evaluating on the TEST split -- final unbiased metric.\n")

    cfg = cfg_from_yaml_file(args.config)
    dcfg = getattr(cfg.dataset, args.split)
    ds = build_dataset_from_cfg(dcfg._base_, dcfg.others)
    print(f":: {args.split}: {len(ds)} frames found.")

    model = builder.model_builder(cfg.model)
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("base_model", state.get("model", state))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"   !! {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"   !! {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    print(f":: loaded {args.ckpt} (epoch {state.get('epoch', '?')})\n")

    model = model.cuda().eval()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mask_ratio is not None:
        model.MAE_encoder.mask_ratio = args.mask_ratio
        print(f":: mask_ratio overridden to {args.mask_ratio}\n")

    rows = []
    step = max(1, len(ds) // args.n)
    for i in range(0, len(ds), step):
        np.random.seed(1000 + i)
        item = ds[i]
        data = item[2]
        pts = data["points"] if isinstance(data, dict) else data
        gt = pts.numpy()
        inp = pts.unsqueeze(0).cuda()

        rec = reconstruct(model, inp, dense_only=args.dense_only,
                    mode=args.mode,
                    n_target=args.n_points if args.n_points
                            else (len(gt) if args.match_density else None))

        # Raw normalised distances for the CDF/histogram figure. Pooled
        # across frames later; per-frame quantiles cannot reconstruct a CDF.
        safe_id = str(item[1]).replace('/', '_').replace('.txt', '').replace('.npy', '')
        d_pg, _ = cKDTree(gt).query(rec, workers=-1)
        d_gp, _ = cKDTree(rec).query(gt, workers=-1)
        _dg, _ = cKDTree(gt).query(gt, k=2, workers=-1)
        _s = _dg[:, 1].mean()
        np.savez_compressed(
            os.path.join(args.out_dir, f'{safe_id}_dists.npz'),
            pg=(d_pg / _s).astype(np.float32),
            gp=(d_gp / _s).astype(np.float32))

        emd = float('nan')
        if not args.no_emd and max(len(rec), len(gt)) <= args.emd_max_points:
            emd = emd_sinkhorn(rec, gt)

        f1, prec, recall = fscore(rec, gt)
        stats = distance_stats(rec, gt)
        rows.append({"id": str(item[1]), "n": len(rec),
                     "cd_l1": chamfer_l1(gt, rec),
                     "cd_l2": chamfer_l2(gt, rec),
                     "spacing": spacing_ratio(rec, gt),
                     "cv": spacing_cv(rec),
                     "cv_gt": spacing_cv(gt),
                     "f1": f1, "prec": prec, "recall": recall,
                     "emd": emd,
                     **stats})

        
        for name, arr, col in (("gt", gt, [1, 0.706, 0]),
                               ("rec", rec, [0, 0.651, 0.929])):
            p = o3d.geometry.PointCloud()
            p.points = o3d.utility.Vector3dVector(arr)
            p.paint_uniform_color(col)
            o3d.io.write_point_cloud(
                os.path.join(args.out_dir, f"{safe_id}_{name}.ply"), p)

    print(f"\n{'Frame ID':<24}{'CD-L1':>12}{'Spacing':>9}{'CV':>7}"
          f"{'F1':>7}{'Prec':>7}{'Rec':>7}")
    print("-" * 73)
    for r in rows:
        print(f"{r['id'][:23]:<24}{r['cd_l1']:>12.4e}{r['spacing']:>9.3f}"
              f"{r['cv']:>7.3f}{r['f1']:>7.3f}{r['prec']:>7.3f}{r['recall']:>7.3f}")
    print("-" * 73)

    def m(k):
        v = np.array([r[k] for r in rows], dtype=float)
        return np.nanmean(v), np.nanstd(v)

    for k, label in (("cd_l1", "CD-L1"), ("cd_l2", "CD-L2"),
                     ("spacing", "Spacing"), ("cv", "CV"), ("cv_gt", "CV_gt"),
                     ("f1", "F1"), ("prec", "Prec"), ("recall", "Recall"),
                     ("pg_med", "P->G med"), ("pg_p90", "P->G p90"),
                     ("pg_p95", "P->G p95"), ("pg_max", "P->G max"),
                     ("gp_med", "G->P med"), ("gp_p90", "G->P p90"),
                     ("gp_p95", "G->P p95"), ("gp_max", "G->P max"),
                     ("emd", "EMD")):
        mu, sd_ = m(k)
        print(f"  mean {label:6s} {mu:.4e}  +/- {sd_:.4e}")
    if np.isnan([r['emd'] for r in rows]).all():
        print(f"  (EMD skipped: > {args.emd_max_points} points, or --no_emd)")
    print(f"\nSaved {len(rows)} pair(s) to ./{args.out_dir}/")
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()