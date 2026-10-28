#!/usr/bin/env python3
"""Row-alignment check across backbones of the same dataset.

Embeddings of the same images from different backbones share similarity
structure: the image-by-image cosine matrices correlate. A backbone whose
rows were written in a different order (or from different images) has no
such correlation with any other backbone, at the level of a shuffled control.

For each dataset, the first N rows of every backbone are compared: Spearman
correlation between upper-triangle cosine similarities, for every pair, plus
a control where one backbone's rows are shuffled.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHEXPERT_BACKBONES,
    MIMIC_BACKBONES,
    Manifest,
    chexpert_paths,
    mimic_paths,
    results_root,
)

N_ROWS = 400


def _sim(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(0)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X @ X.T


def main() -> int:
    from scipy.stats import spearmanr

    out = results_root() / "alignment_check"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "row alignment across backbones (cosine-structure Spearman)")
    man.config = {"n_rows": N_ROWS, "shuffle_seed": 0}
    t0 = time.time()
    iu = np.triu_indices(N_ROWS, 1)
    rows = []
    for ds, bbs, paths in (("chexpert", CHEXPERT_BACKBONES, chexpert_paths),
                           ("mimic", MIMIC_BACKBONES, mimic_paths)):
        sims = {}
        for bb in bbs:
            p = paths(bb)[0]
            man.add_data(p)
            sims[bb] = _sim(np.load(p)[:N_ROWS].astype(np.float64))[iu]
        perm = np.random.default_rng(0).permutation(N_ROWS)
        for a in bbs:
            for b in bbs:
                if a < b:
                    rows.append({"dataset": ds, "a": a, "b": b, "spearman": float(spearmanr(sims[a], sims[b])[0])})
            Xs = np.load(paths(a)[0])[:N_ROWS].astype(np.float64)[perm]
            other = next(b for b in bbs if b != a)
            rows.append({"dataset": ds, "a": a, "b": f"{other} (shuffled {a})",
                         "spearman": float(spearmanr(_sim(Xs)[iu], sims[other])[0])})
    pd.DataFrame(rows).to_csv(out / "alignment.csv", index=False)
    man.time("all", time.time() - t0)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
