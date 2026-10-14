#!/usr/bin/env python3
"""
make_cd_scatter.py -- Chamfer distance against downstream segmentation, one point
per training objective (the six objectives of the downstream table).

All numbers are copied from the paper:
  CD-l1 and CV  -> Table 1 (tab:main), complete-cloud basis, units of 1e-3 for CD
  inferred mIoU -> Table 4 (tab:seg), mean over three segmenter seeds
If a table changes, change OBJECTIVES below; nothing else needs editing.

  python make_cd_scatter.py --out figures/cd_vs_miou.pdf

Writes the PDF for LaTeX and a PNG preview next to it, and prints two checks the
text relies on: that worse CD-l1 orders the objectives by higher inferred mIoU,
and that distance of CV from its target follows the same order.
Requires numpy, matplotlib.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CV_TARGET = 0.571

# name, CD-l1 (1e-3), CV, inferred mIoU, style key, label offset (points), (ha, va)
OBJECTIVES = [
    ("Point-MAE",                        1.698, 1.197, 0.610, "pmae", (5, 0),  ("left", "center")),
    (r"DCD ($\lambda{=}0.5$)",           1.740, 0.930, 0.681, "dcd",  (5, 0),  ("left", "center")),
    (r"Repulsion ($w{=}0.5$)",           1.832, 0.760, 0.737, "rep",  (-5, 0), ("right", "center")),
    (r"DCD ($\lambda{=}1$)",             1.865, 0.688, 0.745, "dcd",  (5, 0),  ("left", "center")),
    ("Ours",                             1.994, 0.606, 0.760, "ours", (0, 5),  ("center", "bottom")),
    (r"Repulsion ($w{=}1$)",             2.112, 0.575, 0.760, "rep",  (0, 5),  ("center", "bottom")),
]

# Point-MAE blue and Ours orange match Fig. 1 (validated pair); the alternatives
# are neutral grey, told apart by marker shape and their direct labels.
STYLE = {"pmae": ("#2a78d6", "o"), "ours": ("#eb6834", "o"),
         "dcd": ("#8a8984", "s"), "rep": ("#8a8984", "^")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def check_orders():
    cd = np.array([o[1] for o in OBJECTIVES])
    cvd = np.abs(np.array([o[2] for o in OBJECTIVES]) - CV_TARGET)
    miou = np.array([o[3] for o in OBJECTIVES])
    by_cd = np.argsort(cd)
    ok_cd = bool(np.all(np.diff(miou[by_cd]) >= 0))
    by_cv = np.argsort(-cvd)
    ok_cv = bool(np.all(np.diff(miou[by_cv]) >= 0))
    print("objective                 CD-l1   |CV-0.571|  inferred mIoU")
    for i in by_cd:
        print(f"  {OBJECTIVES[i][0]:24s}{cd[i]:6.3f}   {cvd[i]:8.3f}     {miou[i]:.3f}")
    print(f"higher CD-l1 -> higher (or equal) inferred mIoU for every step: {ok_cd}")
    print(f"CV closer to target -> higher (or equal) inferred mIoU for every step: {ok_cv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/cd_vs_miou.pdf")
    ap.add_argument("--width", type=float, default=3.3)
    ap.add_argument("--height", type=float, default=1.75)
    a = ap.parse_args()

    check_orders()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    for name, cd, _, miou, key, off, (ha, va) in OBJECTIVES:
        color, marker = STYLE[key]
        ax.plot(cd, miou, ls="", marker=marker, ms=5.5 if marker != "^" else 6,
                mfc=color, mec=SURFACE, mew=0.8, zorder=3)
        ax.annotate(name, (cd, miou), xytext=off, textcoords="offset points",
                    ha=ha, va=va,
                    fontsize=6.5, color=INK)
    ax.set_xlim(1.66, 2.19)
    ax.set_ylim(0.59, 0.785)
    ax.set_xticks([1.7, 1.8, 1.9, 2.0, 2.1])
    ax.set_yticks([0.60, 0.65, 0.70, 0.75])
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(r"CD-$\ell_1$ ($\times10^{-3}$; lower is better)", labelpad=1.5)
    ax.set_ylabel("inferred mIoU (higher is better)", labelpad=2)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=300)
    print(f"\nsaved {a.out} (+ .png preview), {a.width} x {a.height} in")


if __name__ == "__main__":
    main()