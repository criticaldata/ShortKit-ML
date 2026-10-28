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
    DATA,
    MIMIC_BACKBONES,
    Manifest,
    chexpert_paths,
    load_mimic,
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
    from sklearn.metrics import roc_auc_score

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
        if ds == "mimic":
            sims_mimic = sims
        perm = np.random.default_rng(0).permutation(N_ROWS)
        for a in bbs:
            for b in bbs:
                if a < b:
                    rows.append({"dataset": ds, "a": a, "b": b, "spearman": float(spearmanr(sims[a], sims[b])[0])})
            Xs = np.load(paths(a)[0])[:N_ROWS].astype(np.float64)[perm]
            other = next(b for b in bbs if b != a)
            rows.append({"dataset": ds, "a": a, "b": f"{other} (shuffled {a})",
                         "spearman": float(spearmanr(_sim(Xs)[iu], sims[other])[0])})
    # the stored MIMIC ViT-B/32 file (before prepare_mimic_source.py rebuilt it), kept as evidence
    stored = DATA / "mimic_cxr" / "vit32_cls_embeddings.npy"
    probes = []
    if stored.exists():
        man.add_data(stored)
        S = np.load(stored).astype(np.float64)
        s_sim = _sim(S[:N_ROWS])[iu]
        for b in MIMIC_BACKBONES:
            rows.append({"dataset": "mimic", "a": "vit32_cls_stored", "b": b,
                         "spearman": float(spearmanr(s_sim, sims_mimic[b])[0])})
        perm = np.random.default_rng(0).permutation(N_ROWS)
        rows.append({"dataset": "mimic", "a": "vit32_cls_stored", "b": "rad_dino (shuffled vit32_cls_stored)",
                     "spearman": float(spearmanr(_sim(S[:N_ROWS][perm])[iu], sims_mimic["rad_dino"])[0])})
        # plain sex probe (5-fold CV) on the stored and the rebuilt file
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        sex = (load_mimic("vit32_cls")[1].sex == "Male").to_numpy().astype(int)
        for name, X in (("vit32_cls_stored", S), ("vit32_cls", load_mimic("vit32_cls")[0])):
            pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, sex,
                                   cv=5, method="predict_proba")[:, 1]
            probes.append({"dataset": "mimic", "backbone": name, "sex_auroc": float(roc_auc_score(sex, pr))})
    pd.DataFrame(probes).to_csv(out / "alignment_sexprobe.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "alignment.csv", index=False)
    man.time("all", time.time() - t0)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
