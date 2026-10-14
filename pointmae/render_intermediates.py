#!/usr/bin/env python3
"""
render_intermediates.py -- four images of one Point-MAE forward pass, for building
the method diagram yourself:

  1_input_Q.png             input = reference cloud Q, 16,384 points          orange
  2_patches_G512_k64.png    all 512 x 64 patch points: visible orange,
                            masked grey (--patches colours: one colour per patch)
  3_P_hat.png               P-hat, the 307 x 64 masked-patch predictions      blue
  4_output.png              complete output (P-hat + visible patches), 16,384 blue
  5_visible_patches.png     the 205 x 64 visible patch points (encoder input)  orange
  6_masked_patches.png      the 307 x 64 masked patch points (target Q_M)      grey

Input is the intermediates.npz written by dump_intermediates.py. All four use ONE
camera, ONE scale and ONE canvas size, so they line up when you place them side by
side. Orthographic view, z up; points drawn as discs from far to near.

  python render_intermediates.py \
      /project/cvanddl-cea/sgaihre/Point-MAE-baseline/Point-MAE/figures/intermediate/w005_plant_03_0000/intermediates.npz \
      --out_dir /project/cvanddl-cea/sgaihre/Point-MAE-baseline/Point-MAE/figures/intermediate/w005_plant_03_0000/screenshots

  --preview          a sheet of 8 azimuths of Q, to choose --azim
  --azim / --elev    viewing direction in degrees (default 90 / 12)
  --height           image height in pixels (default 1600); point size scales with it
  --radius           point radius in pixels (default: 1.8 per 700 px of height)
  --patches colours  colour each of the 512 patches instead of visible/masked
  --transparent      transparent background instead of white

Requires numpy, pillow.
"""
import argparse
import colorsys
import os

import numpy as np
from PIL import Image, ImageDraw

ORANGE, GREY, BLUE = (235, 104, 52), (168, 167, 162), (42, 120, 214)


def view_basis(azim, elev):
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # towards the camera
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    return right, np.cross(d, right), d


def project(X, basis):
    r, u, d = basis
    return np.c_[X @ r, X @ u], X @ d


def render(layers, basis, bounds, height, radius, transparent=False, ss=3, pad=0.03):
    """layers: list of (points (n,3), colours (n,3) or one rgb); drawn far to near,
    later layers win exact ties."""
    (x0, y0), (x1, y1) = bounds
    marg = pad * (y1 - y0)
    s = (height * ss) / ((y1 - y0) + 2 * marg)
    W = int(round(((x1 - x0) + 2 * marg) * s / ss))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mode, bg = ("RGBA", (255, 255, 255, 0)) if transparent else ("RGB", (255, 255, 255))
    img = Image.new(mode, (W * ss, height * ss), bg)
    dr = ImageDraw.Draw(img)
    xy, depth, col, order = [], [], [], []
    for i, (P, rgb) in enumerate(layers):
        p, z = project(P, basis)
        xy.append(p)
        depth.append(z)
        col.append(np.broadcast_to(np.asarray(rgb, np.uint8).reshape(-1, 3), (len(P), 3)))
        order.append(np.full(len(P), i))
    xy, depth, col, order = map(np.concatenate, (xy, depth, col, order))
    R = radius * ss
    for j in np.lexsort((order, depth)):
        px = (xy[j, 0] - cx) * s + W * ss / 2
        py = height * ss / 2 - (xy[j, 1] - cy) * s
        c = tuple(int(v) for v in col[j]) + ((255,) if transparent else ())
        dr.ellipse((px - R, py - R, px + R, py + R), fill=c)
    return img.resize((W, height), Image.LANCZOS)


def patch_colours(G, seed=0):
    hues = np.random.default_rng(seed).permutation(G) / G
    return np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.65, 0.85)] for h in hues], np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="intermediates.npz from dump_intermediates.py")
    ap.add_argument("--out_dir", default="screenshots")
    ap.add_argument("--azim", type=float, default=90.0)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--patches", choices=["masking", "colours"], default="masking")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=False)
    Q, m, c = z["Q"], z["mask"], z["centres"]
    world = z["patches_local"] + c[:, None, :]                  # (G, k, 3)
    G, k = world.shape[:2]
    P_hat, out = z["P_hat"].reshape(-1, 3), z["complete"]
    os.makedirs(a.out_dir, exist_ok=True)

    if a.preview:
        tiles = []
        for az in range(0, 360, 45):
            b = view_basis(az, a.elev)
            p, _ = project(Q, b)
            im = render([(Q, ORANGE)], b, (p.min(0), p.max(0)), 360, 1.0)
            tile = Image.new("RGB", (360, 380), "white")
            tile.paste(im.resize((min(im.width, 360), 360)), (0, 0))
            ImageDraw.Draw(tile).text((6, 362), f"azim {az}", fill=(0, 0, 0))
            tiles.append(tile)
        sheet = Image.new("RGB", (360 * 4, 380 * 2), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 4) * 360, (i // 4) * 380))
        path = os.path.join(a.out_dir, "preview_azimuths.png")
        sheet.save(path)
        print(f"wrote {path}; pick an azimuth and pass --azim")
        return

    radius = a.radius if a.radius is not None else 1.8 * a.height / 700
    b = view_basis(a.azim, a.elev)
    allp = np.vstack([project(X, b)[0] for X in (Q, world.reshape(-1, 3), P_hat, out)])
    bounds = (allp.min(0), allp.max(0))                         # one frame for all four
    if a.patches == "masking":
        patches = [(world[m].reshape(-1, 3), GREY), (world[~m].reshape(-1, 3), ORANGE)]
    else:
        patches = [(world.reshape(-1, 3), np.repeat(patch_colours(G), k, axis=0))]
    jobs = [("1_input_Q", [(Q, ORANGE)]),
            ("2_patches_G512_k64", patches),
            ("3_P_hat", [(P_hat, BLUE)]),
            ("4_output", [(out, BLUE)]),
            ("5_visible_patches", [(world[~m].reshape(-1, 3), ORANGE)]),
            ("6_masked_patches", [(world[m].reshape(-1, 3), GREY)])]
    print(f"G={G}, k={k}, masked {m.sum()}, visible {(~m).sum()}; azim {a.azim}, elev {a.elev}, "
          f"radius {radius:.1f} px")
    for name, layers in jobs:
        im = render(layers, b, bounds, a.height, radius, a.transparent)
        path = os.path.join(a.out_dir, f"{name}.png")
        im.save(path)
        print(f"  wrote {path}  ({im.width} x {im.height}, {sum(len(L[0]) for L in layers):,} points)")


if __name__ == "__main__":
    main()