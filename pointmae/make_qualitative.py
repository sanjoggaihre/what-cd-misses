#!/usr/bin/env python
"""
make_qualitative.py -- qualitative figures with an identical camera, point
size and point count in every panel.

Three outputs per run:

  fig_qualitative.pdf        plain renders, one colour per cloud
  fig_qualitative_pg.pdf     PRECISION map: predicted points coloured by
                             distance to the nearest ground-truth point
  fig_qualitative_gp.pdf     COVERAGE map: ground-truth points coloured by
                             distance to the nearest predicted point

The two error maps answer different questions and must not be confused. The
precision map asks "is each predicted point on the surface?"; the coverage map
asks "is each part of the surface represented?". Density-aware training gives
ground on the first and gains on the second, so the precision map alone
understates it.

CROPPING
--------
--crop_grid renders a labelled grid over the ground truth so a cell can be
chosen; --crop ROW COL then restricts every panel to that cell, giving a
leaf-level view. Errors are always computed on the FULL clouds and filtered
afterwards, so points near the crop boundary keep their true nearest
neighbour.

Headless-safe (matplotlib only); runs on a cluster node with no display.

Usage:
    python make_qualitative.py --frame plant_03_0080 --preview
    python make_qualitative.py --frame plant_03_0080 --crop_grid
    python make_qualitative.py --frame plant_03_0080 --azim 60 --elev 10 \\
        --crop 0 2 --size 3 --vmax 2.0
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.spatial import cKDTree

ROOT = "/project/cvanddl-cea/sgaihre/cdf_stats"

# (panel title, subdirectory under --root). The ground-truth panel reads the
# *_gt.ply from the first entry; the GT subsample is seeded per frame, so every
# directory's GT file is identical.
PANELS = [
    ("Ground truth",            "w005"),
    ("Point-MAE",               "baseline"),
    ("Ours ($w_{sp}$=0.005)",   "w005"),
    ("Repulsion (h=.007, w=1)", "rep_h007_w1"),
    ("Coordination decoder",    "use_refine_v4"),
    ("DCD",                     "dcd_a20"),
]

GT_COLOUR = "#E69F00"      # orange, marking the reference
PRED_COLOUR = "#0072B2"    # blue; both colour-blind safe
EXPECTED_N = 16384


# ---------------------------------------------------------------- loading
def load_ply(path):
    """Read xyz from a .ply, trying open3d, then plyfile, then ASCII."""
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(path).points)
        if pts.size:
            return pts.astype(np.float64)
    except Exception:
        pass
    try:
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    except Exception:
        pass
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        if not any("format ascii" in h for h in header):
            raise SystemExit(f"{path} is binary PLY and neither open3d nor "
                             "plyfile is installed: pip install plyfile --user")
        n = int(next(h.split()[-1] for h in header
                     if h.startswith("element vertex")))
        data = np.loadtxt(fh, max_rows=n)
    return data[:, :3].astype(np.float64)


def find_ply(root, sub, frame, kind):
    hits = sorted(glob.glob(os.path.join(root, sub, f"*{frame}*_{kind}.ply")))
    if not hits:
        raise SystemExit(f"no *{frame}*_{kind}.ply in {os.path.join(root, sub)}\n"
                         "check --root, the subdirectory name, and --frame")
    return hits[0]


# ---------------------------------------------------------------- geometry
def orient(pts, up):
    """Reorder axes so the plant's vertical axis is plotted as matplotlib z."""
    if up == "z":
        return pts
    if up == "y":
        return pts[:, [0, 2, 1]]
    if up == "x":
        return pts[:, [1, 2, 0]]
    raise ValueError(up)


def axis_limits(pts, pad=0.05, lo=0.0, hi=100.0):
    """Per-axis limits; true proportions are preserved by set_box_aspect.

    A single cube sized by the plant's height wastes most of the panel, since
    plants are tall and narrow.
    """
    mn = np.percentile(pts, lo, axis=0)
    mx = np.percentile(pts, hi, axis=0)
    span = mx - mn
    return mn - span * pad, mx + span * pad


def grid_box(gt, rows, cols, r, c, pad=0.05):
    """Box for cell (r, c) of a rows x cols grid over world x and z; full
    depth in y. Row 0 is the top of the plant."""
    mn, mx = gt.min(0), gt.max(0)
    span = mx - mn
    x0 = mn[0] + span[0] * c / cols
    x1 = mn[0] + span[0] * (c + 1) / cols
    z1 = mx[2] - span[2] * r / rows
    z0 = mx[2] - span[2] * (r + 1) / rows
    px, pz = span[0] / cols * pad, span[2] / rows * pad
    return (np.array([x0 - px, mn[1], z0 - pz]),
            np.array([x1 + px, mx[1], z1 + pz]))


def in_box(pts, box):
    lo, hi = box
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def gt_spacing(gt):
    d, _ = cKDTree(gt).query(gt, k=2, workers=-1)
    return d[:, 1].mean()


# ---------------------------------------------------------------- drawing
def draw(ax, pts, lims, elev, azim, size, colour=None, values=None,
         vmax=2.0, cmap="viridis"):
    kw = dict(s=size, linewidths=0, rasterized=True, depthshade=False)
    if values is None:
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colour, **kw)
    else:
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=values,
                        cmap=cmap, vmin=0.0, vmax=vmax, **kw)
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])
    ax.set_zlim(lims[0][2], lims[1][2])
    try:
        span = np.array([lims[1][i] - lims[0][i] for i in range(3)])
        ax.set_box_aspect(tuple(span / span.max()))
    except AttributeError:
        pass                     # matplotlib < 3.3: aspect is approximate
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return sc


def tidy(fig):
    """3D axes reserve room for decorations we have switched off."""
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.12,
                        wspace=-0.30, hspace=0.02)


def save(fig, save_dir, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(save_dir, f"{stem}.{ext}")
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)


def sel(pts, box):
    """Boolean mask for the crop box, or everything if no crop."""
    return in_box(pts, box) if box is not None else np.ones(len(pts), bool)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--frame", default="plant_03_0080",
                    help="frame id substring, e.g. plant_03_0080")
    ap.add_argument("--up", default="z", choices=["x", "y", "z"])
    ap.add_argument("--elev", type=float, default=10.0)
    ap.add_argument("--azim", type=float, default=60.0)
    ap.add_argument("--size", type=float, default=0.5,
                    help="marker size, identical in every panel "
                         "(use ~3 with --crop)")
    ap.add_argument("--vmax", type=float, default=2.0,
                    help="error colour-scale cap, in GT spacings")
    ap.add_argument("--dcd_own_limits", action="store_true",
                    help="zoom the DCD panel out; ignored when --crop is set")
    ap.add_argument("--preview", action="store_true",
                    help="render the GT from 8 azimuths to choose --azim")
    ap.add_argument("--crop", type=int, nargs=2, metavar=("ROW", "COL"),
                    default=None,
                    help="restrict every panel to one grid cell; row 0 is the "
                         "top of the plant")
    ap.add_argument("--crop_grid", action="store_true",
                    help="render a labelled grid over the GT to choose --crop")
    ap.add_argument("--crop_rows", type=int, default=3)
    ap.add_argument("--crop_cols", type=int, default=3)
    ap.add_argument("--crop_pad", type=float, default=0.05)
    ap.add_argument("--save_dir", default="/project/cvanddl-cea/sgaihre/figures")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150,
                         "savefig.bbox": "tight", "savefig.dpi": 400})

    gt = orient(load_ply(find_ply(args.root, PANELS[0][1], args.frame, "gt")),
                args.up)
    lims = axis_limits(gt)
    s = gt_spacing(gt)
    print(f"GT: {len(gt):,} points, mean NN spacing {s:.5f}")

    # ------------------------------------------------------------ preview
    if args.preview:
        fig = plt.figure(figsize=(10, 5))
        for i, az in enumerate(range(0, 360, 45)):
            ax = fig.add_subplot(2, 4, i + 1, projection="3d")
            draw(ax, gt, lims, args.elev, az, args.size, colour=GT_COLOUR)
            ax.set_title(f"azim={az}", pad=-14)
        tidy(fig)
        save(fig, args.save_dir, "preview")
        print("pick the azimuth showing leaves and stem most clearly, "
              "then re-run without --preview")
        return

    # ---------------------------------------------------------- crop grid
    if args.crop_grid:
        rows, cols = args.crop_rows, args.crop_cols
        mn, mx = gt.min(0), gt.max(0)
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.scatter(gt[:, 0], gt[:, 2], s=0.3, c=GT_COLOUR, linewidths=0,
                   rasterized=True)
        for c in range(cols + 1):
            ax.axvline(mn[0] + (mx[0] - mn[0]) * c / cols, lw=0.5, c="0.5")
        for r in range(rows + 1):
            ax.axhline(mn[2] + (mx[2] - mn[2]) * r / rows, lw=0.5, c="0.5")
        for r in range(rows):
            for c in range(cols):
                b = grid_box(gt, rows, cols, r, c, 0.0)
                cx = (b[0][0] + b[1][0]) / 2
                cz = (b[0][2] + b[1][2]) / 2
                ax.text(cx, cz, f"{r},{c}\n{in_box(gt, b).sum():,}",
                        ha="center", va="center", fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("z (up)")
        ax.set_title("pick a cell, then pass --crop ROW COL")
        save(fig, args.save_dir, "crop_grid")
        return

    box = None
    if args.crop is not None:
        box = grid_box(gt, args.crop_rows, args.crop_cols,
                       args.crop[0], args.crop[1], args.crop_pad)
        lims = box
        print(f"crop cell {tuple(args.crop)}: "
              f"{in_box(gt, box).sum():,} GT points")

    # ------------------------------------------------------------ load all
    clouds = []
    for title, sub in PANELS:
        if title == "Ground truth":
            clouds.append((title, gt, True))
            continue
        pts = orient(load_ply(find_ply(args.root, sub, args.frame, "rec")),
                     args.up)
        if len(pts) != EXPECTED_N:
            print(f"!! {title}: {len(pts):,} points, expected {EXPECTED_N:,}. "
                  "Unmatched cardinality changes apparent density -- use the "
                  "evaluation run that subsampled to 16,384.")
        clouds.append((title, pts, False))

    gt_tree = cKDTree(gt)
    letters = "abcdef"
    nrow = 1 if len(clouds) <= 3 else 2
    ncol = int(np.ceil(len(clouds) / nrow))
    figsize = (7.0, 2.8) if nrow == 1 else (7.0, 4.9)

    # ------------------------------------------- figure 1: plain renders
    fig = plt.figure(figsize=figsize)
    for i, (title, pts, is_gt) in enumerate(clouds):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        panel_lims, label = lims, title
        if title == "DCD" and args.dcd_own_limits and box is None:
            panel_lims = axis_limits(pts, lo=1, hi=99)
            label = title + " (zoomed out)"
        m = sel(pts, box)
        draw(ax, pts[m], panel_lims, args.elev, args.azim, args.size,
             colour=GT_COLOUR if is_gt else PRED_COLOUR)
        ax.set_title(f"({letters[i]}) {label}", pad=-14)
    tidy(fig)
    save(fig, args.save_dir, "fig_qualitative")

    # --------------------------- figure 2: PRECISION map (pred -> GT)
    # Errors are computed on the full clouds, then cropped, so points near the
    # crop boundary keep their true nearest neighbour.
    fig = plt.figure(figsize=(figsize[0], figsize[1] + 0.4))
    sc = None
    print("\nPRECISION  (pred -> GT, GT spacings)")
    print(f"{'panel':<26}{'N':>8}{'median':>9}{'p95':>9}{'max':>9}")
    for i, (title, pts, is_gt) in enumerate(clouds):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        label = title
        m = sel(pts, box)
        if is_gt:
            draw(ax, pts[m], lims, args.elev, args.azim, args.size,
                 colour=GT_COLOUR)
        else:
            d, _ = gt_tree.query(pts, workers=-1)
            e = (d / s)[m]
            print(f"{title:<26}{m.sum():>8,}{np.median(e):>9.3f}"
                  f"{np.percentile(e, 95):>9.3f}{e.max():>9.2f}")
            panel_lims = lims
            if title == "DCD" and args.dcd_own_limits and box is None:
                panel_lims = axis_limits(pts, lo=1, hi=99)
                label = title + " (zoomed out)"
            sc = draw(ax, pts[m], panel_lims, args.elev, args.azim, args.size,
                      values=np.clip(e, 0, args.vmax), vmax=args.vmax)
        ax.set_title(f"({letters[i]}) {label}", pad=-14)
    tidy(fig)
    if sc is not None:
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("predicted point $\\rightarrow$ nearest ground-truth point "
                     f"(GT spacings, capped at {args.vmax:g})")
    save(fig, args.save_dir, "fig_qualitative_pg")

    # ---------------------------- figure 3: COVERAGE map (GT -> pred)
    # Every panel renders identical ground-truth geometry; only the colouring
    # differs, so panels are directly comparable. Panel (a) is covered by
    # definition.
    fig = plt.figure(figsize=(figsize[0], figsize[1] + 0.4))
    sc = None
    gm = sel(gt, box)
    print("\nCOVERAGE  (GT -> pred, GT spacings)")
    print(f"{'panel':<26}{'median':>9}{'p95':>9}{'max':>9}{'frac>1.0':>11}")
    for i, (title, pts, is_gt) in enumerate(clouds):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        if is_gt:
            draw(ax, gt[gm], lims, args.elev, args.azim, args.size,
                 values=np.zeros(int(gm.sum())), vmax=args.vmax)
        else:
            d, _ = cKDTree(pts).query(gt, workers=-1)
            e = (d / s)[gm]
            print(f"{title:<26}{np.median(e):>9.3f}"
                  f"{np.percentile(e, 95):>9.3f}{e.max():>9.2f}"
                  f"{(e > 1.0).mean():>11.3f}")
            sc = draw(ax, gt[gm], lims, args.elev, args.azim, args.size,
                      values=np.clip(e, 0, args.vmax), vmax=args.vmax)
        ax.set_title(f"({letters[i]}) {title}", pad=-14)
    tidy(fig)
    if sc is not None:
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("ground-truth point $\\rightarrow$ nearest predicted point "
                     f"(GT spacings, capped at {args.vmax:g})")
    save(fig, args.save_dir, "fig_qualitative_gp")


if __name__ == "__main__":
    main()