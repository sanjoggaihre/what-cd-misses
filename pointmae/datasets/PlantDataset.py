import os
import re
import json
from glob import glob

import numpy as np
import torch

from .build import DATASETS


@DATASETS.register_module()
class PlantDataset(torch.utils.data.Dataset):
    """
    Pepper-4D loader for Point-MAE pretraining.

    Expects:
        <DATA_PATH>/subset_*/plant_*/frames/day_XXX.txt

    Config keys:
        DATA_PATH   root of Pepper-4D
        npoints     points per sample
        subset      'train' | 'val' | 'test' | 'all'   (split BY PLANT)
        GLOBAL_SCALE (optional) float. If absent it is computed once from the
                    training split and cached to scale.json beside the data.
        CACHE       (optional) bool, default True. Convert .txt to .npy on first
                    read; ~50x faster on subsequent epochs.
        VAL_FRACTION  (optional) float, default 0.15
        TEST_FRACTION (optional) float, default 0.15
        RETURN_DICT (optional) bool, default False. See note at __getitem__.
        SPLIT_MANIFEST (optional) path. Writes which plants landed in which
                    split, so the division is inspectable and citable.
    """

    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.subset = config.subset
        self.cache = getattr(config, "CACHE", True)
        self.return_dict = getattr(config, "RETURN_DICT", False)
        val_frac = getattr(config, "VAL_FRACTION", 0.15)
        test_frac = getattr(config, "TEST_FRACTION", 0.15)

        # --- recursive search -------------------------------------------------
        # A flat glob of '<root>/*.txt' finds nothing here: the frames live three
        # levels down under subset_*/plant_*/frames/.
        pattern = os.path.join(self.root, "**", "frames", "*.txt")
        files = sorted(glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(
                f"No frames found under {pattern}\n"
                f"Check DATA_PATH points at the Pepper-4D root, not at a "
                f"single plant folder."
            )

        # --- group by plant, then split BY PLANT ------------------------------
        # Splitting by frame would put Day 10 of a plant in train and Day 12 of
        # the SAME plant in val. Those two clouds are nearly identical, so the
        # model can memorise the plant and val loss stops measuring
        # generalisation. Split by plant and the val set is genuinely unseen.
        by_plant = {}
        for f in files:
            plant = os.path.basename(os.path.dirname(os.path.dirname(f)))
            subset = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(f))))
            by_plant.setdefault(f"{subset}/{plant}", []).append(f)

        plants = sorted(by_plant)
        rng = np.random.default_rng(0)              # fixed seed: same split every
        order = rng.permutation(len(plants))        # run, and across train/val/test
        n_val = max(1, int(round(val_frac * len(plants))))
        n_test = max(1, int(round(test_frac * len(plants))))
        val_ids = {plants[i] for i in order[:n_val]}
        test_ids = {plants[i] for i in order[n_val:n_val + n_test]}

        if self.subset == "train":
            keys = [p for p in plants if p not in val_ids and p not in test_ids]
        elif self.subset == "val":
            keys = [p for p in plants if p in val_ids]
        elif self.subset == "test":
            keys = [p for p in plants if p in test_ids]
        elif self.subset == "all":
            keys = plants
        else:
            raise ValueError(f"unknown subset {self.subset!r}; expected "
                             f"train/val/test/all")

        self.file_list = [f for k in keys for f in sorted(by_plant[k])]
        self.plants = keys

        days = sorted({self._meta(f)[1] for f in self.file_list})
        print(f"[PlantDataset] {self.subset}: {len(self.file_list)} frames "
              f"from {len(keys)}/{len(plants)} plants; "
              f"days {min(days) if days else '-'}..{max(days) if days else '-'}")
        if days and len(days) > 1 and self.subset == "train":
            n_pairs = sum(len(by_plant[k]) * (len(by_plant[k]) - 1) // 2
                          for k in keys)
            print(f"               {n_pairs:,} (t,t') pairs available for the "
                  f"temporal stage")

        manifest = getattr(config, "SPLIT_MANIFEST", None)
        if manifest:
            rec = {"train": [p for p in plants
                             if p not in val_ids and p not in test_ids],
                   "val": sorted(val_ids), "test": sorted(test_ids),
                   "seed": 0, "val_fraction": val_frac,
                   "test_fraction": test_frac}
            with open(manifest, "w") as fh:
                json.dump(rec, fh, indent=2)
        if self.subset == "train" and len(plants) < 6:
            print("  !! very few plants. With a by-plant split the val set may "
                  "be a single plant, so val loss will be noisy.")

        # --- one global scale for the whole dataset ---------------------------
        self.scale = self._get_scale(config)
        print(f"[PlantDataset] global scale {self.scale:.5f}")

    # ------------------------------------------------------------------ #

    def _get_scale(self, config):
        """
        A SINGLE scale factor shared by every sample.

        Per-sample unit-sphere normalisation (subtract centroid, divide by max
        radius) is standard for ShapeNet and wrong here: it rescales every plant
        to the same size, so a Day-10 plant and a Day-30 plant that differ 2.33x
        in height both come out at 1.0. That difference IS the growth signal --
        the thing this whole project predicts. Measured on synthetic pairs, the
        2.33x ratio collapses to 0.98x.

        A shared constant keeps relative size intact while still bringing
        coordinates into a sane numeric range.
        """
        explicit = getattr(config, "GLOBAL_SCALE", None)
        if explicit:
            return float(explicit)

        cache_file = os.path.join(self.root, "scale.json")
        if os.path.isfile(cache_file):
            with open(cache_file) as f:
                return float(json.load(f)["global_scale"])

        radii = []
        sample = self.file_list[:: max(1, len(self.file_list) // 40)]
        for f in sample:
            pts = self._read(f)
            radii.append(float(np.max(np.linalg.norm(pts - pts.mean(0), axis=1))))
        scale = float(np.percentile(radii, 95))

        try:
            with open(cache_file, "w") as f:
                json.dump({"global_scale": scale,
                           "n_sampled": len(sample)}, f, indent=2)
        except OSError:
            pass
        return scale

    def _read(self, path):
        """Read a frame, caching to .npy on first touch (~50x faster after)."""
        npy = os.path.splitext(path)[0] + ".npy"
        if self.cache and os.path.isfile(npy):
            return np.load(npy)

        with open(path, "r", errors="replace") as f:
            head = [ln for _, ln in zip(range(5), f) if ln.strip()]
        delim = next((d for d in (",", ";", "\t")
                      if all(len(ln.split(d)) > 1 for ln in head)), None)
        toks = head[0].split(delim) if delim else head[0].split()
        try:
            [float(t) for t in toks]
            skip = 0
        except ValueError:
            skip = 1

        data = np.loadtxt(path, delimiter=delim, skiprows=skip, ndmin=2)
        if data.shape[1] < 3:
            raise ValueError(f"{path}: expected >=3 columns, got {data.shape[1]}")
        pts = np.ascontiguousarray(data[:, :3], dtype=np.float32)

        if self.cache:
            try:
                np.save(npy, pts)
            except OSError:
                pass
        return pts

    @staticmethod
    def _meta(path):
        plant = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # Match the DAY specifically. A bare r"(\d+)" grabs the '1' out of the
        # 'A1_' prefix, so every frame of plant_01 would be labelled day 1 --
        # silently destroying the time axis the temporal stage depends on.
        m = re.search(r"Day[_\-]?(\d+)", os.path.basename(path), re.I)
        return plant, (int(m.group(1)) if m else -1)

    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]

        # Do NOT swallow load errors and return zeros. An all-zero cloud passes
        # silently through normalisation and training, producing degenerate
        # gradients that are almost impossible to trace back to a bad file.
        pts = self._read(path)

        n = len(pts)
        if n == 0:
            raise ValueError(f"{path} contains no points")
        replace = n < self.npoints
        choice = np.random.choice(n, self.npoints, replace=replace)
        pts = pts[choice]

        # Centre on the STEM BASE, not the centroid. The centroid sits roughly
        # mid-canopy and drifts upward as the plant fills out, so centroid
        # centring puts the origin somewhere different on the plant each day and
        # leaks growth into a translation.
        base = np.empty(3, dtype=np.float32)
        base[2] = np.percentile(pts[:, 2], 1.0)
        low = pts[pts[:, 2] < np.percentile(pts[:, 2], 15.0)]
        base[:2] = np.median(low[:, :2], axis=0) if len(low) else pts[:, :2].mean(0)

        pts = (pts - base) / self.scale
        pts = torch.from_numpy(np.ascontiguousarray(pts)).float()

        plant, day = self._meta(path)
        taxonomy_id = plant
        model_id = f"{plant}_{day:04d}"

        # Forks differ here, so check YOUR runner_pretrain.py:
        #   points = data.cuda()            -> RETURN_DICT: False
        #   points = data['points'].cuda()  -> RETURN_DICT: True
        # Getting it wrong gives either AttributeError (dict has no .cuda) or
        # "IndexError: too many indices for tensor of dimension 3" (string
        # index into a tensor).
        #
        # Every value here must survive default_collate: tensors, ints and
        # floats become batched tensors, strings become a list of strings.
        # Anything else (None, dict, ndarray of objects) will fail at collate
        # time rather than here, which is a confusing place to debug.
        if self.return_dict:
            return taxonomy_id, model_id, {
                "points": pts,
                "day": int(day),
                "plant": str(plant),
                "scale": float(self.scale),
            }
        return taxonomy_id, model_id, pts