"""Image tracks on the procedural bars-and-patch benchmark (see ``image_data``)."""

from __future__ import annotations

import copy
import time

import numpy as np
from scipy.stats import spearmanr

from .image_data import MARGIN, SIZE, generate_shapes
from .image_models import predict, train_classifier
from .metrics import auroc, parse_detection, worst_and_mean

# rho 0.5 is the null: the patch is present but carries no information. The training
# rho does not decide whether a trained model *uses* the patch (at rho 0.9 one seed may
# rely on it and another not), so ground truth is measured: a dataset is a shortcut
# case when graying the patch flips at least RELIANCE_POSITIVE of the patched eval
# predictions. Shortcut-trained models below that are "ambiguous": left out of AUROC,
# kept in the score-vs-reliance rank correlation.
DETECTION_RHOS = (0.5, 0.9, 0.95, 0.98, 1.0)
RELIANCE_POSITIVE = 0.2
DETECTION = {"n_train": 1500, "n_eval": 400}
# Seeds per training rho. Below rho 1 the shared CNN mostly learns the bar and uses the
# patch only weakly (reliance usually < 0.2), so the clear shortcut cases come mainly
# from rho 1; the partial strengths feed the score-vs-reliance rank correlation.
DETECTION_SEEDS = {
    "full": {0.5: 9, 0.9: 3, 0.95: 2, 0.98: 2, 1.0: 6},
    "quick": {0.5: 2, 0.9: 1, 1.0: 1},
}

MITIGATION_RHOS = (0.9, 0.97)
MITIGATION = {"n_train": 2000, "n_test": 2000}
MITIGATION_SEEDS = {"full": 3, "quick": 1}

# Trained models and eval sets, shared by every submission scored in one process
# (training the CNNs dominates the run time). Each submission gets its own copy of a
# model, so nothing one method does to it reaches the next.
_cache: dict = {}


def patch_reliance(model, ds) -> float:
    """Share of patched images whose prediction flips when the patch is painted gray."""
    r0, r1, c0, c1 = ds.patch_box
    X = ds.images[ds.attribute == 1]
    Xg = X.copy()
    Xg[:, :, r0:r1, c0:c1] = 0.5
    return float((predict(model, X) != predict(model, Xg)).mean())


def _detection_case(rho: float, seed: int):
    key = ("det", rho, seed)
    if key not in _cache:
        tr = generate_shapes(DETECTION["n_train"], rho, seed)
        ev = generate_shapes(DETECTION["n_eval"], rho, seed + 1, corner=tr.corner)
        model = train_classifier(tr.images, tr.task_labels, seed=seed)
        _cache[key] = (ev, model, patch_reliance(model, ev))
    return _cache[key]


def label(rho: float, reliance: float) -> str:
    if rho == 0.5:
        return "null"
    return "shortcut" if reliance >= RELIANCE_POSITIVE else "ambiguous"


def detection_suite(suite: str = "full", seed_offset: int = 0):
    for ri, rho in enumerate(DETECTION_RHOS):
        for s in range(DETECTION_SEEDS[suite].get(rho, 0)):
            seed = seed_offset + 70_000 + 1_000 * ri + 10 * s
            yield (rho, seed, *_detection_case(rho, seed))


def pointing_hits(
    heatmaps: np.ndarray, attribute: np.ndarray, box, dilate: int = MARGIN
) -> list[bool]:
    r0, r1, c0, c1 = box
    r0, c0 = max(r0 - dilate, 0), max(c0 - dilate, 0)
    r1, c1 = min(r1 + dilate, SIZE), min(c1 + dilate, SIZE)
    hits = []
    for h, a in zip(heatmaps, attribute, strict=False):
        if a != 1:
            continue
        r, c = np.unravel_index(int(np.argmax(h)), h.shape)
        hits.append(bool(r0 <= r < r1 and c0 <= c < c1))
    return hits


def score_detection(detect, suite: str = "full", seed_offset: int = 0) -> dict:
    rows, hits, seconds = [], [], []
    for rho, seed, ev, model, reliance in detection_suite(suite, seed_offset):
        t0 = time.perf_counter()
        out = detect(ev.images.copy(), ev.task_labels.copy(), copy.deepcopy(model), seed=seed)
        seconds.append(time.perf_counter() - t0)
        score, flag, extras = parse_detection(out)
        heat = extras.get("heatmaps")
        if heat is not None:
            heat = np.asarray(heat, dtype=float)
            if heat.shape != (len(ev.images), SIZE, SIZE):
                raise ValueError(f"heatmaps must have shape (n, {SIZE}, {SIZE}), got {heat.shape}")
            if label(rho, reliance) == "shortcut":
                hits.extend(pointing_hits(heat, ev.attribute, ev.patch_box))
        rows.append(
            {
                "rho": rho,
                "seed": seed,
                "reliance": round(reliance, 6),
                "label": label(rho, reliance),
                "score": round(score, 6),
                "flag": None if flag is None else bool(flag),
            }
        )

    def pick(lab, key="score"):
        return [r[key] for r in rows if r["label"] == lab]

    metrics = {"auroc": auroc(pick("shortcut"), pick("null"))}
    rho_s = spearmanr([r["score"] for r in rows], [r["reliance"] for r in rows]).statistic
    metrics["reliance_spearman"] = float(rho_s) if np.isfinite(rho_s) else 0.0
    if hits:
        metrics["pointing_game"] = float(np.mean(hits))
    if all(r["flag"] is not None for r in rows):
        metrics["fpr"] = float(np.mean(pick("null", "flag")))
        metrics["tpr"] = (
            float(np.mean(pick("shortcut", "flag"))) if pick("shortcut") else float("nan")
        )
    return {
        "metrics": metrics,
        "details": {
            "n_datasets": len(rows),
            "n_by_label": {lab: len(pick(lab)) for lab in ("shortcut", "ambiguous", "null")},
            "per_dataset": rows,
        },
        "seconds_per_dataset": float(np.median(seconds)),
    }


def _mitigation_case(rho: float, seed: int):
    key = ("mit", rho, seed)
    if key not in _cache:
        tr = generate_shapes(MITIGATION["n_train"], rho, seed)
        te = generate_shapes(MITIGATION["n_test"], 0.5, seed + 1, corner=tr.corner)
        erm = predict(train_classifier(tr.images, tr.task_labels, seed=seed), te.images)
        _cache[key] = (tr, te, worst_and_mean(erm, te.task_labels, te.attribute)[0])
    return _cache[key]


def score_mitigation(fit, suite: str = "full", seed_offset: int = 0) -> dict:
    per, seconds = [], []
    for ri, rho in enumerate(MITIGATION_RHOS):
        for s in range(MITIGATION_SEEDS[suite]):
            seed = seed_offset + 90_000 + 1_000 * ri + 10 * s
            tr, te, erm_wga = _mitigation_case(rho, seed)
            t0 = time.perf_counter()
            model = fit(tr.images.copy(), tr.attribute.copy(), tr.task_labels.copy(), seed=seed)
            if not hasattr(model, "predict"):
                raise ValueError("fit() must return an object with predict(images)")
            pred = np.asarray(model.predict(te.images.copy())).astype(int).ravel()
            seconds.append(time.perf_counter() - t0)
            if pred.shape[0] != len(te.images):
                raise ValueError("predict() returned the wrong number of labels")
            wga, mean, groups = worst_and_mean(pred, te.task_labels, te.attribute)
            per.append(
                {
                    "rho": rho,
                    "seed": seed,
                    "worst_group_acc": wga,
                    "mean_acc": mean,
                    "erm_worst_group_acc": erm_wga,
                    "groups": groups,
                }
            )
    metrics = {
        "worst_group_acc": float(np.mean([r["worst_group_acc"] for r in per])),
        "mean_acc": float(np.mean([r["mean_acc"] for r in per])),
    }
    metrics["gain_over_erm"] = metrics["worst_group_acc"] - float(
        np.mean([r["erm_worst_group_acc"] for r in per])
    )
    return {
        "metrics": metrics,
        "details": {"n_settings": len(per), "per_setting": per},
        "seconds_per_dataset": float(np.median(seconds)),
    }
