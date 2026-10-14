#!/usr/bin/env python3
"""
dump_intermediates.py -- save the point clouds INSIDE one Point-MAE forward pass,
for one frame, as PLY files you can open in the VS Code viewer (or render).

It runs your trained model unchanged and records three of its own sub-modules
with forward hooks, so nothing is re-implemented:
  group_divider  -> patch centres c_i (G=512, by FPS) and patch points (k=64, by kNN)
  MAE_encoder    -> the random mask (60% of patches masked)
  increase_dim   -> the head's k x 3 points per masked patch, in each patch's local frame

Files written to --out_dir (world frame = the evaluation's normalised coordinates):
  00_input_Q.ply                 input = reference cloud Q, 16,384 points        (orange)
  01_centres.ply                 the G patch centres c_i                          (black)
  02_patches_all.ply             all G x k patch memberships, one colour per patch
  03_masking.ply                 visible patches orange, masked patches grey
  04_visible_patches.ply         visible patches only (encoder input)             (orange)
  05_masked_targets_QM.ply       ground truth of the masked patches, Q_M          (grey)
  06_pred_masked_P.ply           P-hat: head output + c_i, masked patches only    (blue)
  07_complete_output.ply         P-hat + visible patches, subsampled to 16,384    (blue)
  intermediates.npz              all of the above as arrays, plus the mask

Run it from the Point-MAE repository root (it imports tools/ and utils/), on a GPU
node, with the same config and checkpoint you evaluated:
  cd /project/cvanddl-cea/sgaihre/Point-MAE-baseline/Point-MAE
  python dump_intermediates.py --config cfgs/<your w005 config>.yaml \
      --gt /project/cvanddl-cea/sgaihre/cdf_stats/w005/plant_03_0080_gt.ply \
      --out_dir /project/cvanddl-cea/sgaihre/intermediates/w005_plant_03_0080
--ckpt defaults to the only ckpt-last.pth under experiments/<config name>/ (the
final checkpoint, as in the paper); pass --ckpt to choose another.

The input is the frame's seeded 16,384-point ground truth written by the
evaluation, so it is in exactly the coordinates the model saw. The script prints
CD-l1 between the complete output and Q; for w005 it should be close to Table 1
(about 2e-3). A value far from that means the coordinates do not match.
The mask is random: --seed picks one draw (same seed -> same mask).

Requires torch and the repository's own dependencies (pointnet2_ops, knn).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ORANGE, GREY, BLUE, BLACK = (235, 104, 52), (185, 184, 179), (42, 120, 214), (20, 20, 20)
_PLY = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
        "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
        "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


# ----------------------------------------------------------------------------- PLY I/O
def read_ply_xyz(path):
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise SystemExit(f"{path}: not a PLY file")
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
            return data[:, [names.index(c) for c in "xyz"]].astype(np.float32)
        dt = np.dtype([(nm, ("<" if fmt == "binary_little_endian" else ">") + t) for nm, t in props])
        arr = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)


def write_ply(path, xyz, rgb=None):
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    n = len(xyz)
    if rgb is None:
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    else:
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rgb = np.broadcast_to(np.asarray(rgb, np.uint8).reshape(-1, 3), (n, 3))
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(arr.tobytes())


# ----------------------------------------------------------------------------- model
def load_model(config_path, ckpt_path):
    try:
        from tools import builder
        from utils.config import cfg_from_yaml_file
    except ImportError as e:
        raise SystemExit(f"cannot import the repository's tools/utils ({e}); "
                         "run this script from the Point-MAE repository root")
    config = cfg_from_yaml_file(config_path)
    model = builder.model_builder(config.model)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:                                   # torch < 1.13 has no weights_only
        state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("base_model", state.get("model", state)) if isinstance(state, dict) else state
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match the config's model:\n  missing {missing[:8]}\n"
                         f"  unexpected {unexpected[:8]}")
    epoch = state.get("epoch") if isinstance(state, dict) else None
    print(f"loaded {ckpt_path}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def find_ckpt(config_path):
    stem = os.path.splitext(os.path.basename(config_path))[0]
    hits = sorted(glob.glob(os.path.join("experiments", stem, "**", "ckpt-last.pth"), recursive=True))
    if len(hits) != 1:
        listing = "\n  ".join(hits) if hits else "(none)"
        raise SystemExit(f"--ckpt not given and experiments/{stem}/ holds {len(hits)} ckpt-last.pth:\n"
                         f"  {listing}\npass --ckpt explicitly")
    return hits[0]


def capture(model, pts):
    """Run the model's own forward once and record group_divider, MAE_encoder and
    increase_dim outputs."""
    rec = {}
    for name in ("group_divider", "MAE_encoder", "increase_dim"):
        if not hasattr(model, name):
            kids = ", ".join(n for n, _ in model.named_children())
            raise SystemExit(f"the model has no sub-module '{name}'. Its sub-modules are: {kids}")

    def hook(name):
        def fn(_m, _inp, out):
            rec.setdefault(name, []).append(out)
        return fn

    handles = [getattr(model, n).register_forward_hook(hook(n))
               for n in ("group_divider", "MAE_encoder", "increase_dim")]
    try:
        with torch.no_grad():
            try:
                model(pts)
            except TypeError:
                model(pts, vis=True)
    finally:
        for h in handles:
            h.remove()
    for n in ("group_divider", "MAE_encoder", "increase_dim"):
        if n not in rec:
            raise SystemExit(f"'{n}' was not called during forward")
        if len(rec[n]) > 1:
            print(f"note: '{n}' ran {len(rec[n])} times; using its first call")
    neighborhood, center = rec["group_divider"][0][:2]
    mask = rec["MAE_encoder"][0][1]
    head = rec["increase_dim"][0]                      # (B, 3k, M) from the Conv1d head
    B, G, k, _ = neighborhood.shape
    M = int(mask[0].sum())
    rebuild = head.transpose(1, 2).reshape(B, M, -1, 3)
    if rebuild.shape[2] != k:
        raise SystemExit(f"head returns {rebuild.shape[2]} points per patch, patches have {k}")
    return (neighborhood[0].float().cpu().numpy(), center[0].float().cpu().numpy(),
            mask[0].bool().cpu().numpy(), rebuild[0].float().cpu().numpy())


def cd_l1(a, b):
    from scipy.spatial import cKDTree
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def patch_colours(G, seed=0):
    import colorsys
    rng = np.random.default_rng(seed)
    hues = rng.permutation(G) / G
    return np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.65, 0.85)] for h in hues], np.uint8)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="default: the only ckpt-last.pth under experiments/<config name>/")
    ap.add_argument("--gt", required=True, help="<frame>_gt.ply written by the evaluation")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seed for the random mask")
    ap.add_argument("--n_out", type=int, default=16384, help="complete output is subsampled to this")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: the grouping (FPS, kNN) needs CUDA; run on a GPU node")
    Q = read_ply_xyz(a.gt)
    print(f"input {a.gt}: {len(Q)} points")
    model = load_model(a.config, a.ckpt or find_ckpt(a.config)).cuda().eval()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    nb, c, m, rb = capture(model, torch.from_numpy(Q)[None].cuda())
    G, k = nb.shape[:2]
    world = nb + c[:, None, :]                           # every patch in the world frame
    P_hat = rb + c[m][:, None, :]                        # head output moved by its centre
    vis, QM = world[~m], world[m]
    full = np.concatenate([vis.reshape(-1, 3), P_hat.reshape(-1, 3)])
    keep = np.random.default_rng(a.seed).choice(len(full), min(a.n_out, len(full)), replace=False)
    out = full[keep]

    print(f"G={G} patches of k={k} ({G * k:,} memberships); masked {m.sum()} ({m.mean():.0%}), "
          f"visible {(~m).sum()}")
    print(f"P-hat: {P_hat.shape[0]} x {k} = {P_hat.shape[0] * k:,} points")
    print(f"CD-l1(P-hat, Q_M)            = {cd_l1(P_hat.reshape(-1, 3), QM.reshape(-1, 3)):.4e}  (training CD term)")
    print(f"CD-l1(complete output, Q)    = {cd_l1(out, Q):.4e}  (compare Table 1; about 2e-3 for w005)")

    os.makedirs(a.out_dir, exist_ok=True)
    pc = patch_colours(G, a.seed)
    files = [
        ("00_input_Q.ply", Q, ORANGE),
        ("01_centres.ply", c, BLACK),
        ("02_patches_all.ply", world.reshape(-1, 3), np.repeat(pc, k, axis=0)),
        ("03_masking.ply", world.reshape(-1, 3), np.repeat(np.where(m[:, None], GREY, ORANGE), k, axis=0)),
        ("04_visible_patches.ply", vis.reshape(-1, 3), ORANGE),
        ("05_masked_targets_QM.ply", QM.reshape(-1, 3), GREY),
        ("06_pred_masked_P.ply", P_hat.reshape(-1, 3), BLUE),
        ("07_complete_output.ply", out, BLUE),
    ]
    for name, xyz, rgb in files:
        write_ply(os.path.join(a.out_dir, name), xyz, rgb)
        print(f"  wrote {name:28s} {len(xyz.reshape(-1, 3)):>6,} points")
    np.savez_compressed(os.path.join(a.out_dir, "intermediates.npz"), Q=Q, centres=c, patches_local=nb,
                        mask=m, head_local=rb, P_hat=P_hat, visible=vis, Q_M=QM, complete=out)
    print(f"  wrote intermediates.npz\nall in {a.out_dir}")


if __name__ == "__main__":
    sys.exit(main())