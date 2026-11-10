"""Embedding tracks: planted attribute signal in synthetic embeddings.

Built on :func:`shortcut_detect.benchmark.conditional_synthetic.generate_conditional_synthetic`.
Every dataset's columns are shuffled with its own seed, so the attribute and
task dimensions sit at different indices each time and cannot be hard-coded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from ..benchmark.conditional_synthetic import generate_conditional_synthetic
from .metrics import auroc, parse_detection, worst_and_mean

VARIANTS = ("linear", "xor", "variance")
DETECTION_DELTAS = (0.0, 0.3, 0.6, 1.0)  # 0.0 is the null
DETECTION = {"n_samples": 600, "embedding_dim": 64, "rho": 0.7, "attribute_dims": 4, "task_dims": 4}
SEEDS = {"full": 10, "quick": 2}
# As many nulls as positives per variant (3 strengths x 10 seeds). With only 10 nulls a
# per-variant AUROC swings by about +-0.15 from seed noise alone (v1.0 had 10).
NULL_SEEDS = {"full": 30, "quick": 2}

MITIGATION_SETTINGS = ((1.0, 0.9), (1.0, 0.97), (2.0, 0.9), (2.0, 0.97))  # (delta, train rho)
MITIGATION = {
    "n_train": 2000,
    "n_test": 4000,
    "embedding_dim": 64,
    "attribute_dims": 4,
    "task_dims": 4,
    "task_effect": 0.8,
}
MITIGATION_SEEDS = {"full": 5, "quick": 1}


@dataclass
class EmbeddingDataset:
    embeddings: np.ndarray
    attribute: np.ndarray
    task_labels: np.ndarray
    delta: float
    variant: str
    seed: int


def _shuffled(ds, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed + 7_919).permutation(ds.embeddings.shape[1])
    return np.ascontiguousarray(ds.embeddings[:, perm])


def detection_suite(suite: str = "full", seed_offset: int = 0):
    for vi, variant in enumerate(VARIANTS):
        for di, delta in enumerate(DETECTION_DELTAS):
            for s in range((NULL_SEEDS if delta == 0 else SEEDS)[suite]):
                seed = seed_offset + 10_000 * (vi + 1) + 100 * di + s
                ds = generate_conditional_synthetic(
                    variant=variant, delta=delta, seed=seed, **DETECTION
                )
                yield EmbeddingDataset(
                    _shuffled(ds, seed), ds.attribute, ds.task_labels, delta, variant, seed
                )


def score_detection(detect, suite: str = "full", seed_offset: int = 0) -> dict:
    rows = []
    seconds = []
    for ds in detection_suite(suite, seed_offset):
        t0 = time.perf_counter()
        out = detect(ds.embeddings.copy(), ds.attribute.copy(), ds.task_labels.copy(), seed=ds.seed)
        seconds.append(time.perf_counter() - t0)
        score, flag, _ = parse_detection(out)
        rows.append((ds.variant, ds.delta, score, flag))

    metrics: dict[str, float] = {}
    per_variant = []
    for v in VARIANTS:
        pos = [r[2] for r in rows if r[0] == v and r[1] > 0]
        neg = [r[2] for r in rows if r[0] == v and r[1] == 0]
        metrics[f"auroc_{v}"] = auroc(pos, neg)
        per_variant.append(metrics[f"auroc_{v}"])
    metrics["auroc"] = float(np.mean(per_variant))
    flags = [r for r in rows if r[3] is not None]
    if flags and len(flags) == len(rows):
        metrics["fpr"] = float(np.mean([r[3] for r in rows if r[1] == 0]))
        metrics["tpr"] = float(np.mean([r[3] for r in rows if r[1] > 0]))
        for d in DETECTION_DELTAS[1:]:
            metrics[f"tpr_delta_{d}"] = float(np.mean([r[3] for r in rows if r[1] == d]))
    details = {
        "n_datasets": len(rows),
        "per_dataset": [
            {"variant": v, "delta": d, "score": round(s, 6), "flag": f} for v, d, s, f in rows
        ],
    }
    return {
        "metrics": metrics,
        "details": details,
        "seconds_per_dataset": float(np.median(seconds)),
    }


# ---------------------------------------------------------------- mitigation


def _mitigation_split(delta: float, rho: float, seed: int):
    common = {
        k: MITIGATION[k] for k in ("embedding_dim", "attribute_dims", "task_dims", "task_effect")
    }
    tr = generate_conditional_synthetic(
        n_samples=MITIGATION["n_train"], delta=delta, rho=rho, variant="linear", seed=seed, **common
    )
    te = generate_conditional_synthetic(
        n_samples=MITIGATION["n_test"],
        delta=delta,
        rho=0.5,
        variant="linear",
        seed=seed + 1,
        **common,
    )
    return (
        EmbeddingDataset(_shuffled(tr, seed), tr.attribute, tr.task_labels, delta, "linear", seed),
        EmbeddingDataset(
            _shuffled(te, seed), te.attribute, te.task_labels, delta, "linear", seed + 1
        ),
    )


def mitigation_suite(suite: str = "full", seed_offset: int = 0):
    for si, (delta, rho) in enumerate(MITIGATION_SETTINGS):
        for s in range(MITIGATION_SEEDS[suite]):
            seed = seed_offset + 50_000 + 1_000 * si + 10 * s
            yield (delta, rho, seed, *_mitigation_split(delta, rho, seed))


def erm_predict(tr: EmbeddingDataset, te_E: np.ndarray) -> np.ndarray:
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(tr.embeddings, tr.task_labels)
    return clf.predict(te_E)


def conditional_leakage(E: np.ndarray, a: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Out-of-fold AUROC of a logistic probe predicting A from [E, Y] (0.5 = no leakage)."""
    X = StandardScaler().fit_transform(np.c_[E, y])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, a * 2 + y)
    p = cross_val_predict(
        LogisticRegression(max_iter=2000), X, a, cv=list(folds), method="predict_proba"
    )
    return float(roc_auc_score(a, p[:, 1]))


def score_mitigation(fit, suite: str = "full", seed_offset: int = 0) -> dict:
    per = []
    seconds = []
    for delta, rho, seed, tr, te in mitigation_suite(suite, seed_offset):
        t0 = time.perf_counter()
        model = fit(tr.embeddings.copy(), tr.attribute.copy(), tr.task_labels.copy(), seed=seed)
        if not hasattr(model, "predict"):
            raise ValueError("fit() must return an object with predict(embeddings)")
        pred = np.asarray(model.predict(te.embeddings.copy())).astype(int).ravel()
        seconds.append(time.perf_counter() - t0)
        if pred.shape[0] != te.embeddings.shape[0]:
            raise ValueError("predict() returned the wrong number of labels")
        wga, mean, groups = worst_and_mean(pred, te.task_labels, te.attribute)
        erm_wga, _, _ = worst_and_mean(erm_predict(tr, te.embeddings), te.task_labels, te.attribute)
        row = {
            "delta": delta,
            "rho": rho,
            "seed": seed,
            "worst_group_acc": wga,
            "mean_acc": mean,
            "erm_worst_group_acc": erm_wga,
            "groups": groups,
        }
        if hasattr(model, "transform"):
            Z = np.asarray(model.transform(te.embeddings.copy()), dtype=float)
            row["leakage_auroc"] = conditional_leakage(Z, te.attribute, te.task_labels, seed)
        per.append(row)

    metrics = {
        "worst_group_acc": float(np.mean([r["worst_group_acc"] for r in per])),
        "mean_acc": float(np.mean([r["mean_acc"] for r in per])),
    }
    metrics["gain_over_erm"] = metrics["worst_group_acc"] - float(
        np.mean([r["erm_worst_group_acc"] for r in per])
    )
    if all("leakage_auroc" in r for r in per):
        metrics["leakage_auroc"] = float(np.mean([r["leakage_auroc"] for r in per]))
    return {
        "metrics": metrics,
        "details": {"n_settings": len(per), "per_setting": per},
        "seconds_per_dataset": float(np.median(seconds)),
    }
