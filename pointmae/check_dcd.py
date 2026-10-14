#!/usr/bin/env python3
"""
check_dcd.py -- compare your DCD implementation against a reference that follows
the official calc_dcd (Wu et al., NeurIPS 2021, arXiv 2111.12702):

  dist1[i]  = squared distance from GT point i to its nearest prediction
  count1[j] = number of GT points whose nearest prediction is j
  weight1   = 1 / (count1[nn(i)]**lambda + 1e-6) * (n_gt / n_x)
  loss1     = mean_i (1 - exp(-alpha * dist1[i]) * weight1[i])
  loss2     = the same with the roles of GT and prediction swapped
  DCD       = (loss1 + loss2) / 2          (one value per batch item)

Both the loss values and the gradients with respect to the prediction are compared.
DCD is symmetric in its two arguments, so the (pred, gt) order does not matter.

Usage (run from your Point-MAE root so that --module is importable):
  python check_dcd.py --module models.uniform_loss --func dcd_loss
--func may be a function f(pred, gt, alpha=..., n_lambda=...) or an nn.Module class
whose __init__ takes alpha / lambda and whose forward takes (pred, gt). Argument
names containing "alpha" or "lambda" are filled in automatically.
"""

import argparse
import importlib
import inspect
import sys

import torch


def dcd_reference(x, gt, alpha=1000.0, n_lambda=1.0):
    """x: predictions (B, N, 3); gt: ground truth (B, M, 3). Returns (B,), float64."""
    x, gt = x.double(), gt.double()
    n_x, n_gt = x.shape[1], gt.shape[1]
    frac_12, frac_21 = n_x / n_gt, n_gt / n_x
    # squared distances written out so the gradient is well defined at zero distance
    d = ((gt ** 2).sum(-1)[:, :, None] + (x ** 2).sum(-1)[:, None, :]
         - 2 * gt @ x.transpose(1, 2)).clamp_min(0)
    dist1, idx1 = d.min(dim=2)                        # each GT point -> nearest prediction
    dist2, idx2 = d.min(dim=1)                        # each prediction -> nearest GT point
    out = []
    for b in range(x.shape[0]):
        c1 = torch.bincount(idx1[b], minlength=n_x)
        w1 = (c1[idx1[b]].double() ** n_lambda + 1e-6) ** -1 * frac_21
        c2 = torch.bincount(idx2[b], minlength=n_gt)
        w2 = (c2[idx2[b]].double() ** n_lambda + 1e-6) ** -1 * frac_12
        l1 = (1 - torch.exp(-alpha * dist1[b]) * w1).mean()
        l2 = (1 - torch.exp(-alpha * dist2[b]) * w2).mean()
        out.append((l1 + l2) / 2)
    return torch.stack(out)


def map_kwargs(sig_params, alpha, lam):
    kw = {}
    for name in sig_params:
        low = name.lower()
        if "alpha" in low or low in ("t", "temp", "temperature"):
            kw[name] = alpha
        elif "lambda" in low or low in ("lam", "n_lam", "lmbda"):
            kw[name] = lam
    return kw


def make_caller(obj, alpha, lam):
    """Return (callable(pred, gt), kwargs used) for a function or an nn.Module class."""
    if inspect.isclass(obj):
        kw = map_kwargs(inspect.signature(obj.__init__).parameters, alpha, lam)
        inst = obj(**kw)
        return (lambda a, b: inst(a, b)), kw
    kw = map_kwargs(inspect.signature(obj).parameters, alpha, lam)
    return (lambda a, b: obj(a, b, **kw)), kw


def first_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def cases():
    g = torch.Generator().manual_seed(0)
    B, N = 2, 2048
    gt = torch.rand(B, N, 3, generator=g)
    sp = 0.05  # rough mean NN spacing of 2048 uniform points in a unit cube
    dup = gt.clone()
    dup[:, N // 2:] = gt[:, : N // 2]                  # half the points duplicated
    yield "identical", gt.clone(), gt
    yield "noise 0.25 spacing", gt + 0.25 * sp * torch.randn(B, N, 3, generator=g), gt
    yield "noise 1 spacing", gt + 1.0 * sp * torch.randn(B, N, 3, generator=g), gt
    yield "half duplicated", dup, gt
    yield "clustered", gt[:, torch.randint(0, N // 8, (N,), generator=g)], gt
    yield "fewer predictions", gt[:, : N // 2] + 0.25 * sp * torch.randn(
        B, N // 2, 3, generator=g), gt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True, help="e.g. models.uniform_loss")
    ap.add_argument("--func", required=True, help="name of your DCD function or class")
    ap.add_argument("--swap", action="store_true",
                    help="pass (gt, pred) instead of (pred, gt); only matters if yours "
                         "is not symmetric")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=1e-4, help="max abs error on the loss")
    ap.add_argument("--gtol", type=float, default=1e-2,
                    help="max gradient error, relative to the largest reference gradient "
                         "(float32 rounding gives ~1e-3 at alpha=1000; real bugs give >0.1)")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    obj = getattr(importlib.import_module(args.module), args.func)
    print(f"Testing {args.module}.{args.func} on {args.device}\n")

    worst_v, worst_g, shown = 0.0, 0.0, False
    for alpha in (50.0, 200.0, 1000.0):
        for lam in (0.5, 1.0):
            call, kw = make_caller(obj, alpha, lam)
            if not shown:
                print(f"argument mapping used: {kw or 'none -- alpha/lambda NOT passed!'}\n")
                shown = True
            for name, pred, gt in cases():
                # reference value and gradient (float64, CPU)
                xr = pred.double().requires_grad_()
                ref = dcd_reference(xr, gt, alpha, lam)
                # yours (float32, chosen device)
                xu = pred.to(args.device).float().requires_grad_()
                g_dev = gt.to(args.device).float()
                got = first_tensor(call(g_dev, xu) if args.swap else call(xu, g_dev))
                scalar = got.numel() == 1
                (ref.mean() if scalar else ref.sum()).backward()
                got.sum().backward()
                ref_v = ref.detach().mean().reshape(1) if scalar else ref.detach()
                got_v = got.detach().double().reshape(-1).cpu()
                err_v = float((got_v - ref_v).abs().max())
                g_ref, g_got = xr.grad, xu.grad.double().cpu()
                scale = float(g_ref.abs().max())
                err_g = float((g_got - g_ref).abs().max()) / max(scale, 1e-12)
                if scale < 1e-9:                     # zero gradient (identical clouds)
                    err_g = float(g_got.abs().max())
                worst_v, worst_g = max(worst_v, err_v), max(worst_g, err_g)
                flag = "" if err_v <= args.tol and err_g <= args.gtol else "   <-- MISMATCH"
                print(f"alpha={alpha:6.0f} lambda={lam:3.1f}  {name:20s} "
                      f"ref {ref_v.mean():.5f}  yours {got_v.mean():.5f}  "
                      f"loss diff {err_v:.1e}  grad diff {err_g:.1e}{flag}")

    ok = worst_v <= args.tol and worst_g <= args.gtol
    print(f"\nLargest loss difference {worst_v:.1e}, largest relative gradient "
          f"difference {worst_g:.1e} -> "
          + ("matches the reference (values and gradients)." if ok else
             "DOES NOT match. Common causes: unsquared distances, lambda applied to the "
             "weight instead of the count, the n_gt/n_x factors swapped or missing, a "
             "missing /2 or a loss weight applied inside the function, or a custom "
             "Chamfer kernel whose backward pass differs."))


if __name__ == "__main__":
    main()