#!/usr/bin/env python3
"""UMAP coordinates for the two illustration figures (needs umap-learn).

1. MIMIC-CXR RAD-DINO, coloured by race. Subjects with no race recorded
   (unknown, unable to obtain, declined) are left out, as in the audit.
2. CelebA ResNet-50, coloured by Male, on a 5,000-image subsample.
Standardised embeddings, default UMAP settings, fixed seed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Manifest, celeba_paths, load_celeba, load_mimic, mimic_paths, results_root  # noqa: E402

SEED = 0
N_CELEBA = 5000


def embed(X: np.ndarray) -> np.ndarray:
    import umap
    from sklearn.preprocessing import StandardScaler

    return umap.UMAP(random_state=SEED).fit_transform(StandardScaler().fit_transform(X))


def main() -> int:
    out = results_root() / "analyses"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "UMAP coordinates for the illustration figures")
    man.config = {"seed": SEED, "celeba_subsample": N_CELEBA, "scaling": "StandardScaler",
                  "umap": "umap-learn defaults (n_neighbors 15, min_dist 0.1)"}
    t0 = time.time()

    X, meta = load_mimic("rad_dino")
    keep = meta.race.notna().to_numpy()
    xy = embed(X[keep])
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "race": meta.race[keep].str.title().to_numpy()}).to_csv(
        out / "umap_mimic_raddino_race.csv", index=False)
    man.add_data(*mimic_paths("rad_dino"))
    man.notes.append(f"MIMIC-CXR: {int((~keep).sum())} subjects without recorded race left out")

    X, attrs = load_celeba()
    rows = np.sort(np.random.default_rng(SEED).choice(len(X), N_CELEBA, replace=False))
    xy = embed(X[rows])
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "male": (attrs.Male.to_numpy()[rows] == 1)}).to_csv(
        out / "umap_celeba_resnet50_male.csv", index=False)
    man.add_data(*celeba_paths())

    man.time("all", time.time() - t0)
    man.out_dir = out / "umap"
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
