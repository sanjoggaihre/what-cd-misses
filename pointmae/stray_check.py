"""Share of predictions farther than 10 tau from the reference (Sec. 5.6).
Usage (GPU node): python stray_check.py w005 dcd_a50 dcd_a50_seed1 dcd_a50_seed2 dcd_a200
"""
import sys, os, glob, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covcheck_test import read_ply_xyz, nn_dist, mean_spacing
D = '/project/cvanddl-cea/sgaihre/cdf_stats'
for m in sys.argv[1:]:
    s = []
    for g in sorted(glob.glob(f'{D}/{m}/*_gt.ply')):
        q = torch.from_numpy(read_ply_xyz(g)).cuda()
        p = torch.from_numpy(read_ply_xyz(g.replace('_gt.ply', '_rec.ply'))).cuda()
        s.append((nn_dist(p, q) / mean_spacing(q) > 10).float().mean().item())
    print(f'{m:16s} beyond 10 tau: {100*sum(s)/len(s):.3f}%  ({len(s)} frames)')
