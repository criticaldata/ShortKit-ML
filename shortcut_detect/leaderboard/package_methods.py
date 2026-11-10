"""Adapters that put ShortKit-ML's own detectors and mitigations on the leaderboard.

Each ``contrib/shortkit-*`` folder is a thin wrapper around one of these functions,
so the leaderboard measures the package's methods exactly as users call them.

Detection: every detector runs through ``ShortcutDetector(methods=[m])`` with the
attribute as ``group_labels``. The **flag** is the package's own decision rule
(``benchmark.method_utils.method_flag``, the one the paper benchmark uses); the
**score** is the continuous statistic that rule thresholds, so AUROC ranks the
same evidence the detector reports.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np


def _run_detector(method: str, embeddings, attribute, task_labels, seed: int) -> dict:
    from shortcut_detect import ShortcutDetector

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        det = ShortcutDetector(methods=[method], seed=seed)
        det.fit(embeddings, task_labels, group_labels=attribute)
    result = det.results_.get(method, {})
    if not result.get("success", False):
        raise RuntimeError(f"{method} did not run: {result.get('error', 'no result')}")
    return result


def _neglog10(p) -> float:
    return float(-np.log10(max(float(p), 1e-300)))


def _report_field(result: dict, name: str) -> float:
    rep = result["report"]
    return float(rep[name] if isinstance(rep, dict) else getattr(rep, name))


# method -> statistic its flag is based on (higher = more evidence of a shortcut)
SCORES = {
    "hbac": lambda r: max(c["purity"] for c in r["report"]["cluster_purities"]),
    "probe": lambda r: float(r["results"]["metrics"]["metric_value"]),
    "statistical": lambda r: max(_neglog10(np.min(p)) for p in r["corrected_pvalues"].values()),
    "geometric": lambda r: max((float(b.effect_size) for b in r["bias_pairs"]), default=0.0),
    "frequency": lambda r: float(r["report"]["metrics"]["probe_accuracy"]),
    "bias_direction_pca": lambda r: _report_field(r, "projection_gap"),
    "sis": lambda r: -float(r["metrics"]["mean_sis_size"]),
    "demographic_parity": lambda r: _report_field(r, "dp_gap"),
    "equalized_odds": lambda r: max(_report_field(r, "tpr_gap"), _report_field(r, "fpr_gap")),
    "groupdro": lambda r: float(
        r["report"]["final"]["avg_acc"] - r["report"]["final"]["worst_group_acc"]
    ),
    "gce": lambda r: _neglog10(r["attribute_test"]["p_value"]),
}


def detect_with(method: str, embeddings, attribute, task_labels, seed: int = 0) -> dict:
    from shortcut_detect.benchmark.method_utils import method_flag

    result = _run_detector(method, embeddings, attribute, task_labels, seed)
    return {"score": float(SCORES[method](result)), "flag": bool(method_flag(method, result))}


# ---------------------------------------------------------------- embedding mitigation


class EmbeddingTransformClassifier:
    """A package debiasing transform followed by logistic regression on the task."""

    def __init__(self, transformer, seed: int):
        from sklearn.linear_model import LogisticRegression

        self.transformer = transformer
        self.clf = LogisticRegression(max_iter=2000, random_state=seed)

    def fit(self, embeddings, task_labels):
        self.clf.fit(self.transform(embeddings), task_labels)
        return self

    def transform(self, embeddings):
        return self.transformer.transform(np.asarray(embeddings, dtype=np.float32))

    def predict(self, embeddings):
        return self.clf.predict(self.transform(embeddings))


# ---------------------------------------------------------------- image mitigation


def locate_shortcut(
    images, attribute, task_labels, *, size: int = 4, dilate: int = 1
) -> np.ndarray:
    """Binary (H, W) mask around the ``size`` square that differs most between attribute groups.

    Mitigation tracks may use group labels. Within one task label the task signal is
    independent of the attribute, so the per-pixel difference of attribute-group means,
    averaged over task labels, isolates what the attribute changes in the image.
    """
    diff = 0.0
    for y in np.unique(task_labels):
        a1 = images[(task_labels == y) & (attribute == 1)]
        a0 = images[(task_labels == y) & (attribute == 0)]
        if len(a1) and len(a0):
            diff = diff + np.abs(a1.mean(0) - a0.mean(0)).sum(0)
    h, w = images.shape[-2:]
    ii = np.pad(np.cumsum(np.cumsum(np.broadcast_to(diff, (h, w)), 0), 1), ((1, 0), (1, 0)))
    win = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    r0, c0 = np.unravel_index(int(np.argmax(win)), win.shape)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[max(r0 - dilate, 0) : r0 + size + dilate, max(c0 - dilate, 0) : c0 + size + dilate] = 1.0
    return mask
