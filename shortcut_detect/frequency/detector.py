"""Embedding-adapted frequency shortcut detector.

This detector operates only in embedding space. It does not localize input-domain
frequency artifacts (e.g., image Fourier bands). Instead, it detects whether class
signals are overly concentrated in a small set of embedding dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from shortcut_detect.detector_base import DetectorBase
from shortcut_detect.probes.sklearn_probe import SKLearnProbe
from shortcut_detect.utils import validate_embeddings_labels


@dataclass(frozen=True)
class FrequencyConfig:
    top_percent: float = 0.05
    tpr_threshold: float = 0.5
    fpr_threshold: float = 0.15
    probe_evaluation: str = "train"  # "train" or "holdout"
    probe_holdout_frac: float = 0.2
    random_state: int = 42


class FrequencyDetector(DetectorBase):
    """Embedding-space detector for concentrated class-separable dimensions.

    Args:
        top_percent: Fraction of top dimensions used to summarize shortcut signature.
        tpr_threshold: Per-class true-positive-rate threshold for shortcut flagging.
        fpr_threshold: Per-class false-positive-rate threshold for shortcut flagging.
        probe_estimator: Optional sklearn classifier. Must expose `fit` and either
            `coef_` (preferred) or be otherwise compatible with class prediction.
        probe_evaluation: "train" or "holdout".
        probe_holdout_frac: Holdout fraction if `probe_evaluation="holdout"`.
        random_state: Seed used for holdout split.
    """

    def __init__(
        self,
        *,
        top_percent: float = 0.05,
        tpr_threshold: float = 0.5,
        fpr_threshold: float = 0.15,
        probe_estimator: BaseEstimator | None = None,
        probe_evaluation: str = "train",
        probe_holdout_frac: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(method="frequency")
        self.config = FrequencyConfig(
            top_percent=float(top_percent),
            tpr_threshold=float(tpr_threshold),
            fpr_threshold=float(fpr_threshold),
            probe_evaluation=str(probe_evaluation),
            probe_holdout_frac=float(probe_holdout_frac),
            random_state=int(random_state),
        )
        self.probe_estimator = probe_estimator
        self.probe_: BaseEstimator | None = None

    def _default_probe(self) -> BaseEstimator:
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=self.config.random_state,
        )

    @staticmethod
    def _compute_class_rates(
        y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
    ) -> dict[str, dict[str, float]]:
        class_rates: dict[str, dict[str, float]] = {}
        for c in classes:
            c_mask = y_true == c
            not_c_mask = ~c_mask
            tp = float(np.sum((y_pred == c) & c_mask))
            fn = float(np.sum((y_pred != c) & c_mask))
            fp = float(np.sum((y_pred == c) & not_c_mask))
            tn = float(np.sum((y_pred != c) & not_c_mask))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            class_rates[str(c)] = {
                "tpr": float(tpr),
                "fpr": float(fpr),
                "support": int(np.sum(c_mask)),
            }
        return class_rates

    @staticmethod
    def _top_dims_from_probe(
        probe: BaseEstimator, classes: np.ndarray, top_k: int
    ) -> dict[str, list[int]]:
        if not hasattr(probe, "coef_"):
            return {}
        coef = np.asarray(probe.coef_)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        classes_arr = np.asarray(classes)
        if classes_arr.ndim != 1 or classes_arr.size < 2:
            return {}

        # sklearn binary logistic regression stores one coefficient row for classes_[1].
        if coef.shape[0] == 1 and classes_arr.size == 2:
            coef_rows = [(classes_arr[1], coef[0])]
        else:
            n_rows = min(coef.shape[0], classes_arr.size)
            coef_rows = [(classes_arr[i], coef[i]) for i in range(n_rows)]

        top_dims: dict[str, list[int]] = {}
        for class_label, coef_row in coef_rows:
            w = np.abs(coef_row)
            order = np.argsort(w)[::-1][:top_k]
            top_dims[str(class_label)] = [int(i) for i in order.tolist()]
        return top_dims

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> FrequencyDetector:
        X, y = validate_embeddings_labels(embeddings, labels, min_samples=10)
        classes = np.unique(y)
        n_classes = int(classes.shape[0])
        if n_classes < 2:
            raise ValueError("FrequencyDetector requires at least two unique classes.")

        if not (0.0 < self.config.top_percent <= 1.0):
            raise ValueError("top_percent must be in (0, 1].")
        if self.config.probe_evaluation not in {"train", "holdout"}:
            raise ValueError("probe_evaluation must be 'train' or 'holdout'.")
        if not (0.0 < self.config.probe_holdout_frac < 1.0):
            raise ValueError("probe_holdout_frac must be in (0, 1).")

        base_probe = (
            self.probe_estimator if self.probe_estimator is not None else self._default_probe()
        )
        probe_runner = SKLearnProbe(
            estimator=base_probe,
            metric="accuracy",
            threshold=1.1,  # frequency uses class-rate thresholds, not SKLearnProbe thresholding
            evaluation=self.config.probe_evaluation,
            test_size=self.config.probe_holdout_frac,
            random_state=self.config.random_state,
        )
        probe_runner.fit(X, y)

        probe = probe_runner.estimator_
        y_true_eval = probe_runner.y_true_eval_
        y_pred = probe_runner.y_pred_eval_
        if probe is None or y_true_eval is None or y_pred is None:
            raise RuntimeError(
                "SKLearnProbe did not produce fitted estimator and evaluation outputs."
            )

        self.probe_ = probe

        class_rates = self._compute_class_rates(y_true_eval, y_pred, classes=np.unique(y_true_eval))
        shortcut_classes = [
            cls
            for cls, m in class_rates.items()
            if np.isfinite(m["tpr"])
            and np.isfinite(m["fpr"])
            and m["tpr"] >= self.config.tpr_threshold
            and m["fpr"] <= self.config.fpr_threshold
        ]

        top_k = max(1, int(np.ceil(self.config.top_percent * X.shape[1])))
        top_dims = self._top_dims_from_probe(probe, classes=classes, top_k=top_k)
        acc = float(accuracy_score(y_true_eval, y_pred))
        cm = confusion_matrix(y_true_eval, y_pred, labels=np.unique(y_true_eval))

        shortcut_detected = len(shortcut_classes) > 0
        frac_shortcut_classes = len(shortcut_classes) / max(1, n_classes)
        if frac_shortcut_classes >= 0.5:
            risk_level = "high"
        elif frac_shortcut_classes > 0.0:
            risk_level = "moderate"
        else:
            risk_level = "low"

        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics={
                "probe_accuracy": acc,
                "top_percent": float(self.config.top_percent),
                "tpr_threshold": float(self.config.tpr_threshold),
                "fpr_threshold": float(self.config.fpr_threshold),
                "n_shortcut_classes": int(len(shortcut_classes)),
                "n_classes": n_classes,
            },
            notes=(
                "Embedding-only frequency adaptation. Detects representational shortcut signatures "
                "from class-separable dimensions; it does not localize input-domain frequencies."
            ),
            metadata={
                "probe": probe.__class__.__name__,
                "probe_evaluation": self.config.probe_evaluation,
                "probe_holdout_frac": (
                    self.config.probe_holdout_frac
                    if self.config.probe_evaluation == "holdout"
                    else None
                ),
                "random_state": self.config.random_state,
                "top_k_dims": int(top_k),
                "n_samples_eval": int(y_true_eval.shape[0]),
                "n_features": int(X.shape[1]),
            },
            report={
                "shortcut_classes": shortcut_classes,
                "class_rates": class_rates,
                "top_dims_by_class": top_dims,
                "confusion_matrix": cm.tolist(),
            },
        )
        self._is_fitted = True
        return self
