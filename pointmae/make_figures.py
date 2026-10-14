#!/usr/bin/env python3
"""
make_figures.py -- the paper's two data figures, rebuilt from the evaluation dumps.

Reads <root>/<model>/<frame>_gt.ply (the seeded 16,384-point ground truth) and
<frame>_rec.ply (the complete-cloud reconstruction). Distances are divided by each
frame's tau (mean ground-truth nearest-neighbour spacing), exactly as in
final_checks.py, so the figure and the numbers in the text come from one definition.

1) Fig. 1 (fig:cdf) -- cumulative error distributions, pooled over all frames
   python make_figures.py cdf --root /project/cvanddl-cea/sgaihre/cdf_stats \
       --out figures/cdf_error.pdf
   Prints the values the text quotes (shares within 0.1, 0.2, 1 and 2 spacings).

2) Fig. 2 (fig:qual) -- one frame, ground truth plus three reconstructions,
   predicted points coloured by distance to the nearest ground-truth point
   python make_figures.py qual --root /project/cvanddl-cea/sgaihre/cdf_stats \
       --out figures/fig_qualitative_pg.pdf
   --frame auto (default) picks the frame whose recall gain (ours over Point-MAE)
   is closest to the median gain over all frames, i.e. a typical frame, not the
   best one. Pass --frame plant_03_0080 (say) to choose one yourself. If the plant
   is not upright, set --up x|y|z (and --flip); turn it with --azim / --elev.
   A PNG preview is written next to the PDF.

Requires numpy, scipy, matplotlib.
"""
import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Palette: two validated categorical slots (blue, orange; CVD and normal-vision
# separation pass on a light surface) and a one-hue blue sequential ramp.
# Fixed slot order (blue, orange, aqua, yellow), validated for adjacent line pairs;
# each also has its own dash pattern, so identity never rests on colour alone.
SERIES = {"baseline": ("#2a78d6", (0, (4, 1.5))), "w005": ("#eb6834", "solid"),
          "dcd_a50": ("#1baf7a", (0, (5, 1.5, 1, 1.5))), "rep_h007_w1": ("#eda100", (0, (1, 1.2)))}
SEQ_BLUE = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GT_GRAY = "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

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


def frame_pairs(root, model):
    out = {}
    for g in sorted(glob.glob(os.path.join(root, model, "*_gt.ply"))):
        r = g[:-len("_gt.ply")] + "_rec.ply"
        if os.path.exists(r):
            out[os.path.basename(g)[:-len("_gt.ply")]] = (g, r)
    if not out:
        raise SystemExit(f"no *_gt.ply / *_rec.ply pairs in {os.path.join(root, model)}")
    return out


def distances(gp, rp):
    G, R = read_ply_xyz(gp), read_ply_xyz(rp)
    tg = cKDTree(G)
    tau = tg.query(G, k=2)[0][:, 1].mean()
    return G, R, tg.query(R)[0] / tau, cKDTree(R).query(G)[0] / tau


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ----------------------------------------------------------------------------- Fig. 1
def run_cdf(a):
    style()
    names = dict(zip(a.models, a.labels))
    pooled = {}
    for m in a.models:
        pg, gp = [], []
        for gpath, rpath in frame_pairs(a.root, m).values():
            _, _, d1, d2 = distances(gpath, rpath)
            pg.append(d1)
            gp.append(d2)
        pooled[m] = (np.sort(np.concatenate(pg)), np.sort(np.concatenate(gp)))
        print(f"{names[m]:10s} frames {len(pg)}  distances per direction {len(pooled[m][0]):,}")

    x = np.linspace(0, a.xmax, 600)
    fig, axes = plt.subplots(1, 2, figsize=(a.width, a.height), sharey=True)
    titles = ["(a) prediction $\\to$ ground truth", "(b) ground truth $\\to$ prediction"]
    for k, ax in enumerate(axes):
        for m in a.models:
            d = pooled[m][k]
            if m not in SERIES:
                raise SystemExit(f"no colour set for model {m}; add it to SERIES")
            color, ls = SERIES[m]
            ax.plot(x, np.searchsorted(d, x, side="right") / len(d),
                    color=color, ls=ls, lw=1.2, label=names[m])
        ax.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
        ax.set_xlim(0, a.xmax)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, a.xmax + 1e-9, 1))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(titles[k], fontsize=7, pad=3)
        ax.set_xlabel("distance (multiples of $\\tau$)", labelpad=1.5)
    axes[0].set_ylabel("cumulative share", labelpad=2)
    if len(a.models) <= 2:
        axes[1].legend(loc="lower right", frameon=False, handlelength=2.2)
        fig.tight_layout(pad=0.3, w_pad=0.6)
    else:                                   # one row above both panels, clear of the curves
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False,
                   handlelength=2.2, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.88))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=300)
    print(f"\nsaved {a.out} (+ .png preview)\n\nvalues quoted in the text:")
    for m in a.models:
        for k, lab in ((0, "pred->GT"), (1, "GT->pred")):
            d = pooled[m][k]
            share = lambda t: np.searchsorted(d, t, side="right") / len(d)
            print(f"  {names[m]:10s} {lab}:  <=0.1 {share(0.1):.3f}  <=0.2 {share(0.2):.3f}  "
                  f"<=1 {share(1.0):.3f}  <=2 {share(2.0):.3f}")


# ----------------------------------------------------------------------------- Fig. 2
def pick_frame(a, pairs_by_model):
    base, ours = a.models[0], a.models[1]
    common = sorted(set(pairs_by_model[base]) & set(pairs_by_model[ours]))
    gains = []
    for f in common:
        r = [np.mean(distances(*pairs_by_model[m][f])[3] <= 1.0) for m in (base, ours)]
        gains.append((f, r[0], r[1], r[1] - r[0]))
    med = np.median([g[3] for g in gains])
    print(f"recall at tau per frame ({base} -> {ours}); median gain {med:+.3f}")
    for f, r0, r1, dg in gains:
        print(f"  {f:22s} {r0:.3f} -> {r1:.3f}  ({dg:+.3f})")
    return min(gains, key=lambda g: abs(g[3] - med))[0]


def to_plot_coords(X, up, flip):
    ax = "xyz".index(up)
    others = [i for i in range(3) if i != ax]
    P = X[:, others + [ax]].copy()
    if flip:
        P[:, 2] *= -1
    return P


def run_qual(a):
    style()
    pairs = {m: frame_pairs(a.root, m) for m in a.models}
    frame = pick_frame(a, pairs) if a.frame == "auto" else a.frame
    for m in a.models:
        if frame not in pairs[m]:
            raise SystemExit(f"frame {frame} missing for model {m}")
    print(f"\nframe: {frame}")

    G = read_ply_xyz(pairs[a.models[0]][frame][0])
    if a.up == "auto":                      # clouds are translated to the stem base:
        frac = (G > 0).mean(0)              # the up axis has nearly all points on one side
        k = int(np.argmax(np.abs(frac - 0.5)))
        up, flip = "xyz"[k], frac[k] < 0.5
        print(f"up axis: {'-' if flip else '+'}{up} (override with --up/--flip)")
    else:
        up, flip = a.up, a.flip
    Gp = to_plot_coords(G, up, flip)
    lo, hi = Gp.min(0), Gp.max(0)
    pad = 0.03 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    panels = [("(a) Ground truth", Gp, None)]
    tau = cKDTree(G).query(G, k=2)[0][:, 1].mean()
    nn_gt = cKDTree(G).query(G, k=2)[0][:, 1]
    print(f"\nthis frame: ground-truth spacing CV {nn_gt.std() / nn_gt.mean():.3f}")
    print(f"{'model':12s} error shares (x tau): <0.1   0.1-1   >1 | recall | stacked: nearest other "
          f"prediction <0.1 tau | spacing CV")
    for m, lab in zip(a.models, a.labels):
        _, R, d_pg, d_gp = distances(*pairs[m][frame])
        panels.append((lab, to_plot_coords(R, up, flip), d_pg))
        nn = cKDTree(R).query(R, k=2)[0][:, 1]
        print(f"{lab:12s}                {np.mean(d_pg < 0.1):.2f}   {np.mean((d_pg >= 0.1) & (d_pg <= 1)):.2f}"
              f"    {np.mean(d_pg > 1):.2f} |  {np.mean(d_gp <= 1):.3f} |   {np.mean(nn / tau < 0.1):.3f}"
              f"                          | {nn.std() / nn.mean():.3f}")
    panels[1:] = [(f"({chr(98 + i)}) {lab}", P, d) for i, (lab, P, d) in enumerate(panels[1:])]

    fig = plt.figure(figsize=(a.width, a.height))
    n = len(panels)
    cb_w, gap = 0.05, 0.005
    pw = (1 - cb_w - 0.02 - gap * (n - 1)) / n
    for i, (title, P, d) in enumerate(panels):
        ax = fig.add_axes([i * (pw + gap), 0.08, pw, 0.92], projection="3d")
        if d is None:
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=a.size, c=GT_GRAY, lw=0,
                       depthshade=False, rasterized=True)
        else:
            sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=a.size, c=np.clip(d, 0, a.vmax),
                            cmap=cmap, vmin=0, vmax=a.vmax, lw=0, depthshade=False,
                            rasterized=True)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        try:
            ax.set_box_aspect(hi - lo, zoom=a.zoom)
        except TypeError:                   # matplotlib < 3.6 has no zoom
            ax.set_box_aspect(hi - lo)
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_axis_off()
        fig.text(i * (pw + gap) + pw / 2, 0.02, title, ha="center", va="bottom",
                 fontsize=7, color=INK)
    cax = fig.add_axes([1 - cb_w - 0.01, 0.2, 0.012, 0.6])
    cb = fig.colorbar(sc, cax=cax, extend="max")
    cb.set_label("error ($\\times\\tau$)", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_linewidth(0.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=600)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200)
    print(f"\nsaved {a.out} (+ .png preview)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cdf")
    c.add_argument("--root", required=True)
    c.add_argument("--models", nargs="+", default=["baseline", "w005"])
    c.add_argument("--labels", nargs="+", default=["Point-MAE", "Ours"])
    c.add_argument("--xmax", type=float, default=4.0)
    c.add_argument("--width", type=float, default=3.3)
    c.add_argument("--height", type=float, default=1.7)
    c.add_argument("--out", default="figures/cdf_error.pdf")
    q = sub.add_parser("qual")
    q.add_argument("--root", required=True)
    q.add_argument("--models", nargs="+", default=["baseline", "w005", "rep_h007_w1"])
    q.add_argument("--labels", nargs="+", default=["Point-MAE", "Ours", "Repulsion"])
    q.add_argument("--frame", default="auto")
    q.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    q.add_argument("--flip", action="store_true")
    q.add_argument("--elev", type=float, default=10)
    q.add_argument("--azim", type=float, default=-60)
    q.add_argument("--vmax", type=float, default=2.0)
    q.add_argument("--size", type=float, default=0.15, help="marker area in pt^2")
    q.add_argument("--zoom", type=float, default=1.2, help="enlarge each panel's plant")
    q.add_argument("--width", type=float, default=6.9)
    q.add_argument("--height", type=float, default=1.9)
    q.add_argument("--out", default="figures/fig_qualitative_pg.pdf")
    a = ap.parse_args()
    if len(a.labels) != len(a.models):
        raise SystemExit("--labels must match --models")
    run_cdf(a) if a.cmd == "cdf" else run_qual(a)


if __name__ == "__main__":
    main()