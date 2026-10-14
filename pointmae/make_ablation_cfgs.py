#!/usr/bin/env python3
"""
make_ablation_cfgs.py -- write the three ablation configs from your "Ours" config.

Only the loss settings change. Everything else (data, schedule, model, mask
ratio, batch) is copied from the config you trained "Ours" (w_sp = 0.005,
w_cov = 0.1) with, so the ablations differ from Ours in one setting each.

  abl_cov_only.yaml    w_sp = 0      w_cov = 0.1   coverage on complete Q
  abl_sp_only.yaml     w_sp = 0.005  w_cov = 0     (no coverage term)
  abl_cov_masked.yaml  w_sp = 0.005  w_cov = 0.1   coverage on masked target Q_M

The script finds the two weight keys by name AND by value (the spacing key
must hold 0.005 and the coverage key 0.1 in your Ours config). It prints the
keys it changed. If it cannot find exactly one of each, pass them yourself
as dotted paths, e.g.  --sp_key model.w_spacing  --cov_key model.w_hole
Keys that are missing from the file (code defaults) are written explicitly.

The masked-target run also needs the small code change described in
README (cov_target). Without that change, abl_cov_masked is identical to Ours.

Usage
  python make_ablation_cfgs.py --ours cfgs/<your w005 config>.yaml --outdir cfgs
"""
import argparse
import copy
import re
import sys

import yaml

SP_NAME = re.compile(r"spac", re.I)
COV_NAME = re.compile(r"(cov|hole)", re.I)
WEIGHT_NAME = re.compile(r"(^w|weight|_w$|lambda)", re.I)


def walk(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from walk(v, path + (k,))
    else:
        yield path, d


def get_parent(d, path):
    for k in path[:-1]:
        if k not in d:
            sys.exit(f"'{k}' not found in the config (path {'.'.join(path)})")
        d = d[k]
    return d


def find_key(cfg, name_re, value):
    hits = []
    for path, v in walk(cfg):
        k = str(path[-1])
        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and name_re.search(k) and WEIGHT_NAME.search(k) \
                and abs(float(v) - value) < 1e-12:
            hits.append(path)
    return hits


def parse_path(s):
    return tuple(s.split("."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="config used for Ours (w_sp=0.005, w_cov=0.1)")
    ap.add_argument("--outdir", default="cfgs")
    ap.add_argument("--sp_key", help="dotted path of the spacing weight, e.g. model.wspacing")
    ap.add_argument("--cov_key", help="dotted path of the coverage weight")
    args = ap.parse_args()

    with open(args.ours) as f:
        cfg = yaml.safe_load(f)

    sp = [parse_path(args.sp_key)] if args.sp_key else find_key(cfg, SP_NAME, 0.005)
    cov = [parse_path(args.cov_key)] if args.cov_key else find_key(cfg, COV_NAME, 0.1)
    if len(sp) != 1 or len(cov) != 1:
        print("Could not identify the weight keys uniquely.")
        print("  spacing candidates :", [".".join(map(str, p)) for p in sp])
        print("  coverage candidates:", [".".join(map(str, p)) for p in cov])
        print("Pass --sp_key and --cov_key (dotted paths) and run again.")
        sys.exit(1)
    sp, cov = sp[0], cov[0]
    for label, path in (("spacing weight key :", sp), ("coverage weight key:", cov)):
        parent = get_parent(cfg, path)
        cur = parent.get(path[-1], "not in file (the model uses its code default)")
        print(label, ".".join(map(str, path)), "=", cur)

    variants = {
        "abl_cov_only":   {"sp": 0.0,   "cov": 0.1, "target": None},
        "abl_sp_only":    {"sp": 0.005, "cov": 0.0, "target": None},
        "abl_cov_masked": {"sp": 0.005, "cov": 0.1, "target": "masked"},
    }
    for name, v in variants.items():
        c = copy.deepcopy(cfg)
        get_parent(c, sp)[sp[-1]] = v["sp"]
        get_parent(c, cov)[cov[-1]] = v["cov"]
        if v["target"]:
            # cov_target sits next to the coverage weight, so the model reads it
            # from the same config object
            get_parent(c, cov)["cov_target"] = v["target"]
        out = f"{args.outdir.rstrip('/')}/{name}.yaml"
        with open(out, "w") as f:
            f.write(f"# generated from {args.ours} by make_ablation_cfgs.py\n")
            yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False)
        extra = f", cov_target = {v['target']}" if v["target"] else ""
        print(f"wrote {out}: w_sp = {v['sp']}, w_cov = {v['cov']}{extra}")


if __name__ == "__main__":
    main()