"""Coverage-tail check on the paper's 23 test frames.

For each test reference cloud Q (16,384 points), build Point-MAE's patches the
same way the model does (farthest-point sampling of 512 centres starting at
index 0, then the 64 nearest points of each centre), mask 307 patches at
random, and measure:

  no_mask   share of Q that lies in no masked patch
  tail_out  share of the worst-covered 10% of Q that lies in no masked patch,
            when the prediction is a perfect reconstruction of Q_M
  cov_tau   L_cov of that perfect reconstruction against the complete Q, in tau
  tau       mean nearest-neighbour spacing of Q

Usage (from the Point-MAE repo, on a GPU node):
  python covcheck_test.py --gt_dir /project/cvanddl-cea/sgaihre/cdf_stats/w005 --masks 5
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch


def read_ply_xyz(path):
    """Minimal PLY reader for x, y, z vertex properties (ascii or binary)."""
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            header.append(line)
            if line == 'end_header':
                break
        fmt = [h for h in header if h.startswith('format')][0].split()[1]
        n = int([h for h in header if h.startswith('element vertex')][0].split()[2])
        props = []
        in_vertex = False
        for h in header:
            if h.startswith('element'):
                in_vertex = h.startswith('element vertex')
            elif h.startswith('property') and in_vertex:
                _, typ, name = h.split()[:3]
                props.append((name, typ))
        tmap = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
                'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
                'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
                'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4'}
        if fmt == 'ascii':
            data = np.loadtxt(f, max_rows=n)
            names = [p[0] for p in props]
            cols = [names.index(c) for c in ('x', 'y', 'z')]
            return data[:, cols].astype(np.float32)
        endian = '<' if fmt == 'binary_little_endian' else '>'
        dt = np.dtype([(name, endian + tmap[typ]) for name, typ in props])
        arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([arr['x'], arr['y'], arr['z']], 1).astype(np.float32)


def nn_dist(a, b, chunk=4096):
    """Distance from each row of a to its nearest row of b (both torch, on GPU)."""
    out = []
    for i in range(0, a.shape[0], chunk):
        out.append(torch.cdist(a[i:i + chunk], b).min(1).values)
    return torch.cat(out)


def mean_spacing(q, chunk=4096):
    out = []
    for i in range(0, q.shape[0], chunk):
        d = torch.cdist(q[i:i + chunk], q)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float('inf')
        out.append(d.min(1).values)
    return torch.cat(out).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_dir', required=True, help='folder with the 23 *_gt.ply test references')
    ap.add_argument('--repo', default='.', help='Point-MAE repo (for utils.misc and knn_cuda)')
    ap.add_argument('--groups', type=int, default=512)
    ap.add_argument('--group_size', type=int, default=64)
    ap.add_argument('--mask_ratio', type=float, default=0.6)
    ap.add_argument('--masks', type=int, default=5, help='random masks per frame')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from utils import misc            # Point-MAE's FPS (pointnet2_ops, starts at index 0)
    from knn_cuda import KNN          # same kNN as the model's Group module

    files = sorted(glob.glob(os.path.join(args.gt_dir, '*_gt.ply')))
    if not files:
        sys.exit(f'no *_gt.ply files in {args.gt_dir}')
    knn = KNN(k=args.group_size, transpose_mode=True)
    n_mask = int(args.mask_ratio * args.groups)
    rng = np.random.default_rng(args.seed)

    rows = []
    print(f'{"frame":<28} {"tau":>9} {"no_mask":>8} {"tail_out":>9} {"cov_tau":>8}')
    for path in files:
        q = torch.from_numpy(read_ply_xyz(path)).cuda().float()
        tau = mean_spacing(q).item()
        xyz = q.unsqueeze(0).contiguous()
        center = misc.fps(xyz, args.groups)                    # 1, G, 3
        _, idx = knn(xyz, center)                              # 1, G, k
        idx = idx[0]                                           # G, k
        k_tail = int(0.1 * q.shape[0])
        per = []
        for _ in range(args.masks):
            masked = torch.from_numpy(rng.permutation(args.groups)[:n_mask]).cuda()
            in_m = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
            in_m[idx[masked].reshape(-1)] = True               # points in at least one masked patch
            q_m = q[in_m]                                      # perfect reconstruction of Q_M
            e = nn_dist(q, q_m)                                # e(q; P) for every q in Q
            tail = e.topk(k_tail).indices
            per.append(((~in_m).float().mean().item(),
                        (~in_m[tail]).float().mean().item(),
                        (e[tail].mean() / tau).item()))
        no_mask, tail_out, cov_tau = np.mean(per, 0)
        rows.append((tau, no_mask, tail_out, cov_tau))
        name = os.path.basename(path).replace('_gt.ply', '')
        print(f'{name:<28} {tau:9.5f} {no_mask:8.3f} {tail_out:9.3f} {cov_tau:8.2f}')

    r = np.array(rows)
    print('-' * 66)
    print(f'{len(files)} frames x {args.masks} masks')
    print(f'mean tau            {r[:, 0].mean():.5f}')
    print(f'share in no masked  mean {r[:, 1].mean():.3f}   min {r[:, 1].min():.3f}   max {r[:, 1].max():.3f}')
    print(f'tail outside mask   mean {r[:, 2].mean():.3f}   min {r[:, 2].min():.3f}')
    print(f'L_cov (perfect Q_M) mean {r[:, 3].mean():.2f} tau   min {r[:, 3].min():.2f}   max {r[:, 3].max():.2f}')


if __name__ == '__main__':
    main()