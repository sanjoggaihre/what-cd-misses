"""
Losses that can actually see clumping and holes.

Chamfer distance assigns every predicted point its nearest ground-truth point
with no penalty for collisions, so duplicated points are free. It also averages
the recall term, so a hole spanning 2% of the surface moves the number by
roughly 2% of one term. Neither failure mode is visible to CD, which is why
architectural coordination alone does not fix them.

  spacing_loss  -- penalises predicted points closer together than the GT
                   spacing at that cardinality. GT-referenced, so uniform
                   clumping is caught (self-referential spacing statistics
                   score a uniformly clumped cloud as clean).
  hole_loss     -- mean of the worst q-quantile of GT->pred distances.
  dcd           -- density-aware Chamfer (Wu et al., NeurIPS 2021).
                   Reported and used for checkpoint selection.
"""
import torch

_EPS = 1e-8


@torch.no_grad()
def _knn_idx(pts, k, chunk=4096):
    B, N, _ = pts.shape
    out = []
    for s in range(0, N, chunk):
        d = torch.cdist(pts[:, s:s + chunk], pts)
        out.append(d.topk(k, dim=-1, largest=False).indices)
    return torch.cat(out, dim=1)


@torch.no_grad()
def _nn_index(a, b, chunk=1024):
    """For each point in a, the index of its nearest point in b."""
    out = []
    for s in range(0, a.shape[1], chunk):
        out.append(torch.cdist(a[:, s:s + chunk], b).argmin(dim=-1))
    return torch.cat(out, dim=1)


def _gather_xyz(pts, idx):
    """pts (B,N,3), idx (B,M) or (B,M,k) -> matching xyz, keeps grad."""
    if idx.dim() == 2:
        return pts.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
    B, M, k = idx.shape
    flat = idx.reshape(B, M * k, 1).expand(-1, -1, 3)
    return pts.gather(1, flat).reshape(B, M, k, 3)


def _dist_to(a, b, idx):
    """Differentiable distance from a to its assigned points in b.

    The assignment is piecewise constant, so treating idx as a constant is the
    correct subgradient -- this is exactly what the CUDA Chamfer op does.
    """
    nb = _gather_xyz(b, idx)
    return torch.sqrt(((a.unsqueeze(2) if idx.dim() == 3 else a) - nb)
                      .pow(2).sum(-1) + _EPS)

def repulsion_loss(pred, k=5, h=0.020):
    """PU-Net-style repulsion component (Yu et al., CVPR 2018).

    For every predicted point, the loss evaluates

        phi(d) = -d * exp(-d^2 / h^2)

    over its k nearest predicted neighbours and returns the mean.

    The potential has its minimum at d* = h / sqrt(2). Under gradient
    descent, neighbour distances smaller than d* are pushed apart, while
    distances larger than d* experience an attractive force. However, the
    magnitude of that attractive force decays exponentially and becomes
    negligible for widely separated points.

    This loss uses one global scale h for the complete plant and does not
    adapt to local ground-truth density. In contrast, the proposed spacing
    loss uses a location-dependent reference derived from nearby GT points
    and explicitly penalizes distances outside a tolerance band.

    Note: _dist_to() must return Euclidean distances, not squared distances,
    for this expression to match the PU-Net formulation.
    """
    idx = _knn_idx(pred, k + 1)[:, :, 1:]
    d = _dist_to(pred, pred, idx)
    return (-d * torch.exp(-(d ** 2) / (h ** 2))).mean()


@torch.no_grad()
def gt_spacing(gt):
    """Mean nearest-neighbour distance of the GT, per sample. (B,)"""
    idx = _knn_idx(gt, 2)
    nb = _gather_xyz(gt, idx[:, :, 1])
    return (nb - gt).norm(dim=-1).mean(dim=1).clamp_min(1e-6)

def gt_spacing_local(gt, k=8):
    """Per-point spacing, not a plant-wide mean.

    A single scalar is dominated by leaf surfaces, where most points are.
    Applying it to a stem -- locally much denser in the GT -- pushes those
    points apart until the tube breaks, which is what produced the visible
    stem gaps.
    """
    idx = _knn_idx(gt, k + 1)[:, :, 1:]
    return _dist_to(gt, gt, idx).mean(-1).clamp_min(1e-6)          # (B, M)

# def repulsion_loss(pred, k=5, h=0.028):
#     """PU-Net repulsion term (Yu et al., CVPR 2018), adapted.

#     The canonical uniformity prior for point-cloud upsampling, included as a
#     baseline for the proposed term. Two differences matter:

#       - One-sided. Points closer than the decay radius repel; nothing
#         penalises being too far apart.
#       - Globally parameterised. A single radius h for the whole cloud, with
#         no reference to the ground truth's local spacing -- so leaf laminae
#         and stems receive the same target.

#     h defaults to ~2x the dataset's mean GT nearest-neighbour spacing.
#     Returns a negative value; it is minimised by increasing separation.
#     """
#     idx = _knn_idx(pred, k + 1)[:, :, 1:]
#     d = _dist_to(pred, pred, idx)
#     w = torch.exp(-(d ** 2) / (h ** 2))
#     return -(d * w).mean()


def spacing_loss(pred, gt, k=4, band=0.25):
    """Hold each predicted point to the local spacing of its nearest GT point.

    Two-sided. The one-sided version (penalising only "too close") measured
    spacing 1.38 and CV 0.35 against a GT CV of 0.57: the model satisfied
    repulsion by spreading onto a regular lattice, more uniform than real
    scan data and mostly off-surface (precision 0.39 vs the baseline's 0.84).
    Penalising "too far" as well removes that escape route.
    """
    r_gt = gt_spacing_local(gt)
    near = _nn_index(pred, gt)
    r = r_gt.gather(1, near).unsqueeze(-1)                          # (B, N, 1)

    idx = _knn_idx(pred, k + 1)[:, :, 1:]
    d = _dist_to(pred, pred, idx)                                   # (B, N, k)

    too_close = torch.relu(r * (1 - band) - d) / r
    too_far = torch.relu(d - r * (1 + band)) / r
    return (too_close ** 2 + too_far ** 2).mean()


def hole_loss(pred, gt, q=0.1):
    """Mean of the worst q-quantile of GT->pred distances.

    A hole is a contiguous set of GT points far from any prediction, so it
    dominates this quantile. The plain CD recall term averages it away.
    """
    idx = _nn_index(gt, pred)
    d = _dist_to(gt, pred, idx)                    # (B,M)
    k = max(1, int(q * d.shape[1]))
    return d.topk(k, dim=1).values.mean()


def dcd(pred, gt, alpha=1000.0, n_lambda=1.0):
    """Density-aware Chamfer Distance.

    Defaults reproduce the authors' evaluation configuration:
        alpha = 1000
        n_lambda = 1

    _dist_to() must return Euclidean distance, because it is squared here.
    """
    B, N, _ = pred.shape
    B_gt, M, _ = gt.shape
    assert B == B_gt

    # Nearest-neighbour assignments
    i_pg = _nn_index(pred, gt)       # each prediction -> nearest GT
    i_gp = _nn_index(gt, pred)       # each GT -> nearest prediction

    # Published DCD uses squared L2 distances
    d_pg2 = _dist_to(pred, gt, i_pg).square()
    d_gp2 = _dist_to(gt, pred, i_gp).square()

    # Number of predictions assigned to every GT point
    count_gt = torch.zeros(
        B, M, device=pred.device, dtype=d_pg2.dtype
    )
    count_gt.scatter_add_(
        1, i_pg, torch.ones_like(d_pg2)
    )

    density_pg = (
        count_gt.gather(1, i_pg)
        .detach()
        .pow(n_lambda)
    )

    weight_pg = (
        density_pg.add(1e-6).reciprocal()
        * (float(N) / float(M))
    )

    loss_pg = (
        1.0 - torch.exp(-alpha * d_pg2) * weight_pg
    ).mean(dim=1)

    # Number of GT points assigned to every prediction
    count_pred = torch.zeros(
        B, N, device=pred.device, dtype=d_gp2.dtype
    )
    count_pred.scatter_add_(
        1, i_gp, torch.ones_like(d_gp2)
    )

    density_gp = (
        count_pred.gather(1, i_gp)
        .detach()
        .pow(n_lambda)
    )

    weight_gp = (
        density_gp.add(1e-6).reciprocal()
        * (float(M) / float(N))
    )

    loss_gp = (
        1.0 - torch.exp(-alpha * d_gp2) * weight_gp
    ).mean(dim=1)

    return 0.5 * (loss_pg + loss_gp).mean()