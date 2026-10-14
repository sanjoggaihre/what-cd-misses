#!/usr/bin/env python3
"""
render_views.py -- one compact PNG per point cloud, every model from the same camera.

For one test frame it renders the ground truth and each model's complete-cloud
reconstruction, all in one plain colour, into separate PNG files. Every cloud is
drawn with the same camera, the same axis limits (taken from the ground truth) and
the same point size, then all images are cropped with ONE shared box (the union
of the clouds' extents plus a small margin). So the images are tight around the
plant, the same size, and at the same scale: placed side by side, they line up.

Reads <root>/<model>/<frame>_gt.ply and <frame>_rec.ply (the evaluation dumps).

Example:
  python render_views.py --root /project/cvanddl-cea/sgaihre/cdf_stats \
      --models baseline w005 rep_h007_w1 dcd_a50 --out_dir renders

  --frame auto (default) picks the frame whose recall gain (second model over the
  first) is closest to the median gain, i.e. a typical frame; or pass a name such
  as --frame plant_03_0080.
  Camera: --up x|y|z (+ --flip) if the plant is not upright; --azim / --elev to turn it.
  Zoom: --crop X0 Y0 X1 Y1 also saves the same region of every image (fractions of
        the full image), e.g. one leaf; raise --res for more pixels in the crop.
  Look: --color (one hex colour for every cloud), --size (point size),
        --res (render resolution), --margin (pixels kept around the plant),
        --transparent (no white background).

--mode coverage draws, for each model, the GROUND TRUTH: blue where one of the
model's inferred points (copied input removed) lies within tau, light grey where
none does. This is the generated-output recall of Table 2 made visible.

Writes <out_dir>/<frame>_gt.png and <out_dir>/<frame>_<model>.png, and prints per
model: recall at tau, the share of points stacked within 0.1 tau of another
prediction, and the spacing CV (numbers for the caption; nothing is drawn from them).

Requires numpy, scipy, matplotlib, pillow.
"""
import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PLY = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
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
                props.append((tok[2], _PLY[tok[1]]))
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


def pair(root, model, frame):
    g = os.path.join(root, model, f"{frame}_gt.ply")
    r = os.path.join(root, model, f"{frame}_rec.ply")
    if not (os.path.exists(g) and os.path.exists(r)):
        raise SystemExit(f"missing {frame}_gt.ply or {frame}_rec.ply for model {model}")
    return g, r


def frames_of(root, model):
    return {os.path.basename(p)[:-len("_gt.ply")]
            for p in glob.glob(os.path.join(root, model, "*_gt.ply"))}


def recall_at_tau(G, R):
    tau = cKDTree(G).query(G, k=2)[0][:, 1].mean()
    return np.mean(cKDTree(R).query(G)[0] <= tau)


def pick_frame(root, m0, m1):
    common = sorted(frames_of(root, m0) & frames_of(root, m1))
    if not common:
        raise SystemExit(f"no frames shared by {m0} and {m1}")
    rows = []
    for f in common:
        G = read_ply_xyz(pair(root, m0, f)[0])
        r0 = recall_at_tau(G, read_ply_xyz(pair(root, m0, f)[1]))
        r1 = recall_at_tau(G, read_ply_xyz(pair(root, m1, f)[1]))
        rows.append((f, r0, r1, r1 - r0))
    med = np.median([r[3] for r in rows])
    print(f"recall at tau per frame ({m0} -> {m1}); median gain {med:+.3f}")
    for f, r0, r1, d in rows:
        print(f"  {f:22s} {r0:.3f} -> {r1:.3f}  ({d:+.3f})")
    return min(rows, key=lambda r: abs(r[3] - med))[0]


def to_plot(X, up, flip):
    k = "xyz".index(up)
    P = X[:, [i for i in range(3) if i != k] + [k]].copy()
    if flip:
        P[:, 2] *= -1
    return P


def render_rgba(P, lo, hi, a, colors=None):
    """Render one cloud on a transparent canvas; return an RGBA uint8 array."""
    side = a.res / 100.0                    # square canvas, a.res pixels at dpi 100
    fig = plt.figure(figsize=(side, side), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.patch.set_alpha(0)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=a.size, c=a.color if colors is None else colors,
               lw=0, depthshade=False)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo, zoom=1.0)
    except TypeError:                       # matplotlib < 3.6 has no zoom
        ax.set_box_aspect(hi - lo)
    ax.view_init(elev=a.elev, azim=a.azim)
    ax.set_axis_off()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", default=["baseline", "w005", "rep_h007_w1"])
    ap.add_argument("--frame", default="auto")
    ap.add_argument("--out_dir", default="renders")
    ap.add_argument("--color", default="#2a78d6", help="one colour for every cloud (default blue)")
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--elev", type=float, default=10)
    ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--size", type=float, default=5.0, help="marker area in pt^2")
    ap.add_argument("--res", type=int, default=1600, help="render canvas side, pixels")
    ap.add_argument("--margin", type=int, default=12, help="pixels kept around the plant")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--mode", default="plain", choices=["plain", "coverage"],
                    help="plain: each model's output in one colour. coverage: the ground "
                         "truth, blue where the model's inferred (non-copied) points come "
                         "within tau, light grey where they do not")
    ap.add_argument("--uncovered", default="#d3d2ce", help="colour of uncovered points")
    ap.add_argument("--crop", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                    help="also save the same zoomed region of every image, as fractions of "
                         "the full image (0 0 = top-left, 1 1 = bottom-right), e.g. 0 0.25 0.5 0.6")
    ap.add_argument("--preview", action="store_true",
                    help="only write preview_angles.png: the ground truth from 24 camera angles")
    a = ap.parse_args()

    if a.frame == "auto" and len(a.models) < 2:
        raise SystemExit("--frame auto needs at least two models")
    frame = pick_frame(a.root, a.models[0], a.models[1]) if a.frame == "auto" else a.frame
    print(f"\nframe: {frame}")

    G = read_ply_xyz(pair(a.root, a.models[0], frame)[0])
    for m in a.models[1:]:                  # every model must share this exact ground truth
        Gm = read_ply_xyz(pair(a.root, m, frame)[0])
        if Gm.shape != G.shape or not np.allclose(Gm, G):
            raise SystemExit(f"ground truth of {m} differs from {a.models[0]} for {frame}")
    if a.up == "auto":                      # clouds sit on the stem base: the up axis has
        frac = (G > 0).mean(0)              # nearly all points on one side of zero
        k = int(np.argmax(np.abs(frac - 0.5)))
        up, flip = "xyz"[k], bool(frac[k] < 0.5)
        print(f"up axis: {'-' if flip else '+'}{up} (override with --up / --flip)")
    else:
        up, flip = a.up, a.flip

    # shared limits: the ground truth's box, widened to contain every reconstruction
    clouds = [("gt", G)] + [(m, read_ply_xyz(pair(a.root, m, frame)[1])) for m in a.models]
    allp = np.vstack([to_plot(X, up, flip) for _, X in clouds])
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.02 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad

    tg = cKDTree(G)
    nn_gt = tg.query(G, k=2)[0][:, 1]
    tau = nn_gt.mean()
    print(f"\n{'cloud':14s} {'points':>7s} {'recall':>7s} {'stacked':>8s} {'CV':>7s}   "
          f"(stacked = nearest other point < 0.1 tau)")
    print(f"{'ground truth':14s} {len(G):7d} {'':>7s} {np.mean(nn_gt / tau < 0.1):8.3f} "
          f"{nn_gt.std() / nn_gt.mean():7.3f}")
    for m, R in clouds[1:]:
        nn = cKDTree(R).query(R, k=2)[0][:, 1]
        rec = np.mean(cKDTree(R).query(G)[0] <= tau)
        print(f"{m:14s} {len(R):7d} {rec:7.3f} {np.mean(nn / tau < 0.1):8.3f} "
              f"{nn.std() / nn.mean():7.3f}")

    if a.preview:                           # contact sheet of camera angles, then stop
        os.makedirs(a.out_dir, exist_ok=True)
        elevs, azims = [15, 35, 55], list(range(-180, 180, 45))
        res, size, cell = a.res, a.size, 320
        a.res, a.size = 500, a.size / 3       # thumbnails: smaller canvas, finer points
        sheet = Image.new("RGB", (cell * len(azims), (cell + 24) * len(elevs)), "white")
        draw = ImageDraw.Draw(sheet)
        Gp = to_plot(G, up, flip)
        for i, el in enumerate(elevs):
            for j, az in enumerate(azims):
                a.elev, a.azim = el, az
                im = render_rgba(Gp, lo, hi, a)
                rr, cc = np.where(im[..., 3] > 0)
                im = Image.fromarray(im[rr.min():rr.max() + 1, cc.min():cc.max() + 1])
                im.thumbnail((cell - 10, cell - 10))
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im).convert("RGB")
                x, y = j * cell + (cell - im.size[0]) // 2, i * (cell + 24)
                sheet.paste(im, (x, y + (cell - im.size[1]) // 2))
                draw.text((j * cell + 8, y + cell + 4), f"--azim {az} --elev {el}", fill=(40, 40, 40))
        a.res, a.size = res, size
        path = os.path.join(a.out_dir, "preview_angles.png")
        sheet.save(path)
        print(f"\nsaved {path}: pick an angle, then rerun without --preview using that --azim / --elev")
        return

    # pass 1: render everything; pass 2: crop all with one shared box
    if a.mode == "plain":
        imgs = [(name, render_rgba(to_plot(X, up, flip), lo, hi, a)) for name, X in clouds]
    else:                                   # coverage of the ground truth by inferred points
        Gp = to_plot(G, up, flip)
        blue = np.array(matplotlib.colors.to_rgba(a.color))
        grey = np.array(matplotlib.colors.to_rgba(a.uncovered))
        imgs = [("gt", render_rgba(Gp, lo, hi, a))]
        print(f"\n{'model':14s} {'copied':>7s} {'inferred recall':>16s}   "
              f"(copied = output point within 0.001 tau of a ground-truth point)")
        for m, R in clouds[1:]:
            copied = tg.query(R)[0] < 1e-3 * tau
            inf = R[~copied]
            cov = cKDTree(inf).query(G)[0] <= tau if len(inf) else np.zeros(len(G), bool)
            print(f"{m:14s} {copied.mean():7.3f} {cov.mean():16.3f}")
            order = np.argsort(cov)          # draw uncovered first, covered on top
            cols = np.where(cov[order, None], blue, grey)
            imgs.append((m, render_rgba(Gp[order], lo, hi, a, cols)))
    mask = np.zeros(imgs[0][1].shape[:2], bool)
    for _, im in imgs:
        mask |= im[..., 3] > 0
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise SystemExit("nothing was drawn; check --up / --elev / --azim")
    r0, r1 = max(rows.min() - a.margin, 0), min(rows.max() + a.margin + 1, mask.shape[0])
    c0, c1 = max(cols.min() - a.margin, 0), min(cols.max() + a.margin + 1, mask.shape[1])

    os.makedirs(a.out_dir, exist_ok=True)
    print()
    for name, im in imgs:
        crop = Image.fromarray(im[r0:r1, c0:c1])
        if not a.transparent:
            bg = Image.new("RGBA", crop.size, (255, 255, 255, 255))
            crop = Image.alpha_composite(bg, crop).convert("RGB")
        path = os.path.join(a.out_dir, f"{frame}_{name}.png")
        crop.save(path)
        print(f"saved {path}  ({crop.size[0]} x {crop.size[1]} px)")
        if a.crop:                          # identical zoom box for every cloud
            w, h = crop.size
            x0, y0, x1, y1 = a.crop
            box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            zpath = os.path.join(a.out_dir, f"{frame}_{name}_crop.png")
            crop.crop(box).save(zpath)
            print(f"saved {zpath}  (zoom box {box})")


if __name__ == "__main__":
    main()