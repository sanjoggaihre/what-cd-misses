# dcd_check.py
import torch, sys
sys.path.insert(0, '.')
from models.uniform_loss import dcd, gt_spacing

torch.manual_seed(0)
gt = torch.rand(2, 8192, 3, device='cuda') * 0.5
s = gt_spacing(gt)
print(f'gt spacing {s.mean().item():.6f}')

for mult in (5.0, 20.0, 50.0):
    print(f'\nalpha_mult={mult}  alpha={1/(mult*s.mean().item()):.1f}')
    # perfect
    print(f'  identical      {dcd(gt, gt, alpha_mult=mult).item():.4f}')
    # small noise -- should be low but non-zero
    for scale in (0.25, 1.0, 4.0):
        n = gt + torch.randn_like(gt) * s.mean().item() * scale
        print(f'  noise {scale:>4}x sp  {dcd(n, gt, alpha_mult=mult).item():.4f}')
    # duplicated: half the points stacked -- DCD must punish this
    dup = torch.cat([gt[:, :4096], gt[:, :4096]], dim=1)
    print(f'  50% duplicated {dcd(dup, gt, alpha_mult=mult).item():.4f}')
    # random cloud -- should be near 1.0
    print(f'  random         {dcd(torch.rand_like(gt)*0.5, gt, alpha_mult=mult).item():.4f}')