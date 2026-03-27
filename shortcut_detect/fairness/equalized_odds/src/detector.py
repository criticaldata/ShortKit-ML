"""Equalized Odds detector for shortcut/fairness analysis.

Implements the Hardt et al. (2016) diagnostic of checking true positive and
false positive rate parity across protected groups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ....detector_base import DetectorBase, RiskLevel


@dataclass
class EqualizedOddsReport:
    """Container for group metrics/gaps."""

    group_metrics: dict[str, dict[str, float]]
    tpr_gap: float
    fpr_gap: float
    overall_accuracy: float
    reference: str
    risk_level: str
    notes: str


class EqualizedOddsDetector(DetectorBase):
    """Compute TPR/FPR gaps across demographic groups."""

    def __init__(
        self,
        estimator: LogisticRegression | None = None,
        min_group_size: int = 10,
        tpr_gap_threshold: float = 0.1,
        fpr_gap_threshold: float = 0.1,
    ) -> None:
        super().__init__(method="equalized_odds")

        self.estimator = estimator or LogisticRegression(max_iter=1000)
        self.min_group_size = min_group_size
        self.tpr_gap_threshold = tpr_gap_threshold
        self.fpr_gap_threshold = fpr_gap_threshold

        self.group_metrics_: dict[str, dict[str, float]] = {}
        self.tpr_gap_: float = float("nan")
        self.fpr_gap_: float = float("nan")
        self.overall_accuracy_: float = float("nan")
        self.report_: EqualizedOddsReport | None = None

    def fit(
        self, embeddings: np.ndarray, labels: np.ndarray, group_labels: np.ndarray
    ) -> EqualizedOddsDetector:
        """Train simple classifier and compute equalized odds gaps."""
        if group_labels is None:
            raise ValueError("EqualizedOddsDetector requires group_labels.")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D (n_samples, embedding_dim).")
        if labels.ndim != 1:
            raise ValueError("Labels must be 1D.")
        if group_labels.ndim != 1:
            raise ValueError("group_labels must be 1D.")

        if embeddings.shape[0] != labels.shape[0] or labels.shape[0] != group_labels.shape[0]:
            raise ValueError("Embeddings, labels, and group_labels must align.")

        unique_labels = np.unique(labels)
        if unique_labels.size != 2:
            raise ValueError("Equalized odds requires binary labels.")

        self.estimator.fit(embeddings, labels)
        preds = self.estimator.predict(embeddings)
        self.overall_accuracy_ = float(accuracy_score(labels, preds))

        self.group_metrics_ = self._compute_group_metrics(labels, preds, group_labels)
        self.tpr_gap_ = self._compute_gap("tpr")
        self.fpr_gap_ = self._compute_gap("fpr")

        risk_level, notes = self._assess_risk()
        self.report_ = EqualizedOddsReport(
            group_metrics=self.group_metrics_,
            tpr_gap=self.tpr_gap_,
            fpr_gap=self.fpr_gap_,
            overall_accuracy=self.overall_accuracy_,
            reference="Hardt et al. 2016",
            risk_level=risk_level,
            notes=notes,
        )
        self._finalize_results()
        self._is_fitted = True
        return self

    def _compute_group_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, group_labels: np.ndarray
    ) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for group in np.unique(group_labels):
            mask = group_labels == group
            if mask.sum() < self.min_group_size:
                metrics[str(group)] = {
                    "tpr": float("nan"),
                    "fpr": float("nan"),
                    "support": float(mask.sum()),
                }
                continue

            group_true = y_true[mask]
            group_pred = y_pred[mask]

            tp = float(((group_true == 1) & (group_pred == 1)).sum())
            fn = float(((group_true == 1) & (group_pred == 0)).sum())
            fp = float(((group_true == 0) & (group_pred == 1)).sum())
            tn = float(((group_true == 0) & (group_pred == 0)).sum())

            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

            metrics[str(group)] = {
                "tpr": tpr,
                "fpr": fpr,
                "support": float(mask.sum()),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        return metrics

    def _compute_gap(self, metric_key: str) -> float:
        values = [
            m[metric_key] for m in self.group_metrics_.values() if not np.isnan(m[metric_key])
        ]
        if len(values) < 2:
            return float("nan")
        return float(np.max(values) - np.min(values))

    def _assess_risk(self) -> tuple[str, str]:
        tpr_gap = self.tpr_gap_
        fpr_gap = self.fpr_gap_

        gaps = [gap for gap in [tpr_gap, fpr_gap] if not np.isnan(gap)]
        if not gaps:
            return RiskLevel.UNKNOWN.value, "Insufficient data to assess equalized odds."

        high = any(gap >= 2 * max(self.tpr_gap_threshold, self.fpr_gap_threshold) for gap in gaps)
        moderate = any(gap >= max(self.tpr_gap_threshold, self.fpr_gap_threshold) for gap in gaps)

        if high:
            return RiskLevel.HIGH.value, "Large TPR/FPR disparities detected across groups."
        if moderate:
            return (
                RiskLevel.MODERATE.value,
                "Moderate disparity detected in equalized odds metrics.",
            )
        return RiskLevel.LOW.value, "Equalized odds gaps within tolerance."

    def _finalize_results(self) -> None:
        risk_enum = RiskLevel.from_string(self.report_.risk_level if self.report_ else None)
        risk_level = risk_enum.value
        if self.report_ is not None:
            self.report_.risk_level = risk_level
        if risk_enum in {RiskLevel.HIGH, RiskLevel.MODERATE}:
            shortcut_detected = True
        elif risk_enum == RiskLevel.LOW:
            shortcut_detected = False
        else:
            shortcut_detected = None

        self.shortcut_detected_ = shortcut_detected
        metrics = {
            "tpr_gap": self.tpr_gap_,
            "fpr_gap": self.fpr_gap_,
            "overall_accuracy": self.overall_accuracy_,
        }
        metadata = {
            "min_group_size": self.min_group_size,
            "tpr_gap_threshold": self.tpr_gap_threshold,
            "fpr_gap_threshold": self.fpr_gap_threshold,
        }
        report = asdict(self.report_) if self.report_ else {}
        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics=metrics,
            notes=self.report_.notes if self.report_ else "",
            metadata=metadata,
            report=report,
        )

    def get_report(self) -> dict[str, Any]:
        return super().get_report()
