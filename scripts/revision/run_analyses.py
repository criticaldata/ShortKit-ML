#!/usr/bin/env python3
"""Revision analyses that need new computation on the embeddings.

1. Unsupervised clusters against metadata (R1-Q5.3, R1-Q7.0): k-means (k = 8)
   on each backbone's standardised embeddings, cross-tabulated against view,
   AP/PA, Support Devices, demographics and diagnoses (NMI, Cramer's V, chi2).
2. Group-size stability (R2-19): the non-White group is subsampled (whole
   patients on CheXpert) to m images, White kept whole; conditional probe
   AUROC and rejection rate over repeats.
3. Held-out LDA for MIMIC-CXR RAD-DINO sex: LDA fitted on training folds, test
   projections saved for the figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHEXPERT_BACKBONES,
    CHEXPERT_JOIN_NOTE,
    DATA,
    MIMIC_BACKBONES,
    Manifest,
    chexpert_paths,
    load_chexpert,
    load_mimic,
    mimic_paths,
    results_root,
)

K_CLUSTERS = 8


def mimic_view(meta: pd.DataFrame) -> np.ndarray:
    dicom = meta["image_path"].str.extract(r"/([^/]+)\.dcm$")[0]
    md = pd.read_csv(DATA / "mimic_cxr" / "mimic-cxr-2.0.0-metadata.csv", usecols=["dicom_id", "ViewPosition"])
    return pd.DataFrame({"dicom_id": dicom}).merge(md, on="dicom_id", how="left")["ViewPosition"].fillna("NA").to_numpy()


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    from scipy.stats import chi2_contingency

    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if min(table.shape) < 2:
        return float("nan"), float("nan")
    chi2, p, _, _ = chi2_contingency(table)
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), float(p)


def cluster_vs_metadata(dataset: str, bb: str, X: np.ndarray, cols: dict[str, np.ndarray]) -> list[dict]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
    cl = KMeans(n_clusters=K_CLUSTERS, n_init=4, random_state=0).fit_predict(Z)
    rows = []
    for name, v in cols.items():
        v = np.asarray(v).astype(str)
        tab = pd.crosstab(cl, v).to_numpy()
        cv, p = cramers_v(tab)
        rows.append({"dataset": dataset, "backbone": bb, "variable": name,
                     "nmi": float(normalized_mutual_info_score(v, cl)), "cramers_v": cv, "chi2_p": p,
                     "n_levels": int(len(np.unique(v))),
                     "largest_cluster_share": float(np.bincount(cl).max() / len(cl))})
    return rows


def chexpert_metadata_columns(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    def lab(c):  # positive vs not positive (negative, uncertain or blank)
        return np.where(meta[c].to_numpy() == 1, "pos", "not_pos")

    return {
        "view (frontal/lateral)": meta["view"].to_numpy(),
        "AP/PA": meta["ap_pa"].to_numpy(),
        "Support Devices": lab("Support Devices"),
        "sex": meta["sex"].to_numpy(),
        "age 60+": (meta["age"] >= 60).astype(int).to_numpy(),
        "race White": (meta["race"] == "WHITE").astype(int).to_numpy(),
        "Pneumothorax": lab("Pneumothorax"),
        "Pleural Effusion": lab("Pleural Effusion"),
        "No Finding": lab("No Finding"),
    }


def group_size_stability(dataset, X, meta, sizes, reps, K, threads):
    import torch

    torch.set_num_threads(threads)
    from shortcut_detect.audit import AuditConfig, conditional_audit

    y = meta["task_label"].to_numpy().astype(int)
    race = meta["race"].astype(str).to_numpy()
    a = np.where(race == "WHITE", "White", "non-White")
    grp = meta["patient_id"].to_numpy() if "patient_id" in meta.columns else None
    rows = []
    minority = np.flatnonzero(a == "non-White")
    for m in sizes:
        for r in range(reps):
            rng = np.random.default_rng(10_000 * m + r)
            if grp is not None:  # draw whole patients until m images
                pats = rng.permutation(np.unique(grp[minority]))
                keep, tot = [], 0
                for p_ in pats:
                    idx = minority[grp[minority] == p_]
                    keep.append(idx)
                    tot += len(idx)
                    if tot >= m:
                        break
                sub = np.concatenate(keep)
            else:
                sub = rng.choice(minority, size=min(m, len(minority)), replace=False)
            rows_ = np.sort(np.concatenate([np.flatnonzero(a == "White"), sub]))
            g = None if grp is None else grp[rows_]
            res = conditional_audit(X[rows_], a[rows_], y[rows_],
                                    AuditConfig(n_permutations=K, n_bootstrap=0, detectors=("probe",), seed=r),
                                    groups=g).results["probe"]
            rows.append({"dataset": dataset, "minority_images": int(len(sub)), "target_m": m, "rep": r,
                         "minority_Y1": int(y[sub].sum()), "probe_auroc": res.statistic,
                         "p_value": res.p_value, "reject": bool(res.p_value < 0.05)})
    return rows


def heldout_lda(X, sex, seed=0):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    proj = np.zeros(len(sex))
    fold = np.zeros(len(sex), dtype=int)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, sex)):
        lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        lda.fit(X[tr], sex[tr])
        proj[te] = lda.decision_function(X[te])
        fold[te] = f
    return proj, fold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    out = results_root() / "analyses"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(out, "clusters vs metadata, group-size stability, held-out LDA")
    man.config = {"k_clusters": K_CLUSTERS, "group_size_reps": args.reps, "K": args.K,
                  "group_size_backbone": "rad_dino", "lda": "StandardScaler + shrinkage LDA, 5-fold"}
    man.notes.append(CHEXPERT_JOIN_NOTE)
    t0 = time.time()
    rows = []
    for bb in CHEXPERT_BACKBONES:
        X, meta = load_chexpert(bb)
        man.add_data(chexpert_paths(bb)[0])
        rows += cluster_vs_metadata("chexpert", bb, X, chexpert_metadata_columns(meta))
    for bb in MIMIC_BACKBONES:
        X, meta = load_mimic(bb)
        man.add_data(*mimic_paths(bb))
        cols = {"view (AP/PA)": mimic_view(meta), "sex": meta["sex"].to_numpy(),
                "age 60+": (meta["age"] >= 60).astype(int).to_numpy(),
                "race White": (meta["race"] == "WHITE").astype(int).to_numpy(),
                "mortality": meta["task_label"].to_numpy()}
        rows += cluster_vs_metadata("mimic", bb, X, cols)
    man.add_data(DATA / "mimic_cxr" / "mimic-cxr-2.0.0-metadata.csv")
    pd.DataFrame(rows).to_csv(out / "cluster_metadata.csv", index=False)
    man.time("clusters", time.time() - t0)

    t1 = time.time()
    Xc, mc = load_chexpert("rad_dino")
    Xm, mm = load_mimic("rad_dino")
    jobs = [("chexpert", Xc, mc, [s]) for s in (25, 50, 100, 200, 400)]
    jobs += [("mimic", Xm, mm, [s]) for s in (25, 50, 100, 200, 400)]
    parts = Parallel(n_jobs=args.jobs)(
        delayed(group_size_stability)(d, X, m, s, args.reps, args.K, args.threads) for d, X, m, s in jobs
    )
    pd.DataFrame([r for p in parts for r in p]).to_csv(out / "group_size_stability.csv", index=False)
    man.time("group_size", time.time() - t1)

    proj, fold = heldout_lda(Xm, (mm["sex"] == "Male").astype(int).to_numpy())
    pd.DataFrame({"projection": proj, "fold": fold, "sex": mm["sex"].to_numpy()}).to_csv(
        out / "lda_heldout_mimic_raddino_sex.csv", index=False)
    man.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
