"""Reference-cloud checks quoted in Sec. 3.4.

On the 23 test ground-truth clouds (16,384 points each) this reports:
  - share of points with an exact duplicate (paper: 1.4%)
  - relative change in median r(q) after removing duplicates (paper: 0.5%)
  - mean 4-NN distance as a multiple of r(q) (paper: 0.75)
  - L_sp(Q, Q) with 4 predicted neighbours, b = 0.25 (paper: 0.033)

Usage (from the repo root):
  python downstream/gt_checks.py /project/cvanddl-cea/sgaihre/cdf_stats/baseline
"""
import os
import sys, glob, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pointmae', 'models'))
from uniform_loss import spacing_loss, gt_spacing_local, _knn_idx, _dist_to

def read_ply(p):
    with open(p,'rb') as f:
        h=b''
        while not h.endswith(b'end_header\n'): h+=f.readline()
        dt=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('r','u1'),('g','u1'),('b','u1')])
        a=np.fromfile(f,dtype=dt,count=16384)
    return np.stack([a['x'],a['y'],a['z']],1)

files=sorted(glob.glob(sys.argv[1]+'/*_gt.ply'))
print(len(files),'frames')
rows=[]
for fp in files:
    q=read_ply(fp)
    _,inv,cnt=np.unique(q,axis=0,return_inverse=True,return_counts=True)
    dup=(cnt[inv.ravel()]>1).mean()
    Q=torch.tensor(q,dtype=torch.float32)[None]
    r=gt_spacing_local(Q,8)[0]
    qu=torch.tensor(np.unique(q,axis=0),dtype=torch.float32)[None]
    ru=gt_spacing_local(qu,8)[0]
    # spacing loss of reference against itself, paper setting and matched k
    l4=spacing_loss(Q,Q,k=4,band=0.25).item()
    l8=spacing_loss(Q,Q,k=8,band=0.25).item()
    # fraction of 4-NN ratios outside band, and mean ratio
    idx=_knn_idx(Q,5)[:,:,1:]; d=_dist_to(Q,Q,idx)[0]/r[:,None]
    out=((d<0.75)|(d>1.25)).float().mean().item()
    rows.append([dup, r.median().item(), ru.median().item(), l4, l8, out, d.mean().item()])
a=np.array(rows)
names=['dup_frac','r_med','r_med_dedup','Lsp_k4','Lsp_k8','frac_out_band_k4','mean_ratio_k4']
for n,m in zip(names,a.mean(0)): print(f'{n:20s} {m:.5f}')
print('rel change r median (dedup vs not):', ((a[:,2]-a[:,1])/a[:,1]).mean())
