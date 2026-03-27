"""Demographic Parity detector for shortcut/fairness analysis.

Implements Feldman et al. (2015) demographic parity gap (difference in
positive prediction rates across protected groups).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from ....detector_base import DetectorBase


@dataclass
class DemographicParityReport:
    """Container for demographic parity metrics."""

    group_rates: dict[str, dict[str, float]]
    dp_gap: float
    overall_positive_rate: float
    reference: str
    risk_level: str
    notes: str


class DemographicParityDetector(DetectorBase):
    """Compute demographic parity gap across demographic groups."""

    def __init__(
        self,
        estimator: LogisticRegression | None = None,
        min_group_size: int = 10,
        dp_gap_threshold: float = 0.1,
    ) -> None:
        super().__init__(method="demographic_parity")

        self.estimator = estimator or LogisticRegression(max_iter=1000)
        self.min_group_size = min_group_size
        self.dp_gap_threshold = dp_gap_threshold

        self.group_rates_: dict[str, dict[str, float]] = {}
        self.dp_gap_: float = float("nan")
        self.overall_positive_rate_: float = float("nan")
        self.report_: DemographicParityReport | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> DemographicParityDetector:
        """Train classifier and compute demographic parity gap."""
        if group_labels is None:
            raise ValueError("DemographicParityDetector requires group_labels.")

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
            raise ValueError("Demographic parity requires binary labels.")

        self.estimator.fit(embeddings, labels)
        preds = self.estimator.predict(embeddings)
        positive_label = self.estimator.classes_[1]
        self.overall_positive_rate_ = float(np.mean(preds == positive_label))

        self.group_rates_ = self._compute_group_rates(preds, group_labels, positive_label)
        self.dp_gap_ = self._compute_gap()

        risk_level, notes = self._assess_risk()
        self.report_ = DemographicParityReport(
            group_rates=self.group_rates_,
            dp_gap=self.dp_gap_,
            overall_positive_rate=self.overall_positive_rate_,
            reference="Feldman et al. 2015",
            risk_level=risk_level,
            notes=notes,
        )
        self._finalize_results()
        self._is_fitted = True
        return self

    def _compute_group_rates(
        self, y_pred: np.ndarray, group_labels: np.ndarray, positive_label: float
    ) -> dict[str, dict[str, float]]:
        rates: dict[str, dict[str, float]] = {}
        for group in np.unique(group_labels):
            mask = group_labels == group
            support = float(mask.sum())
            if support < self.min_group_size:
                rates[str(group)] = {"positive_rate": float("nan"), "support": support}
                continue
            positive_rate = float(np.mean(y_pred[mask] == positive_label))
            rates[str(group)] = {"positive_rate": positive_rate, "support": support}
        return rates

    def _compute_gap(self) -> float:
        values = [
            m["positive_rate"]
            for m in self.group_rates_.values()
            if not np.isnan(m["positive_rate"])
        ]
        if len(values) < 2:
            return float("nan")
        return float(np.max(values) - np.min(values))

    def _assess_risk(self) -> tuple[str, str]:
        gap = self.dp_gap_
        if np.isnan(gap):
            return "unknown", "Insufficient data to assess demographic parity."

        if gap >= 2 * self.dp_gap_threshold:
            return "high", "Large demographic parity gap detected across groups."
        if gap >= self.dp_gap_threshold:
            return "moderate", "Moderate demographic parity gap detected."
        return "low", "Demographic parity gap within tolerance."

    def _finalize_results(self) -> None:
        risk_level = self.report_.risk_level if self.report_ else "unknown"
        if risk_level in {"high", "moderate"}:
            shortcut_detected = True
        elif risk_level == "low":
            shortcut_detected = False
        else:
            shortcut_detected = None

        self.shortcut_detected_ = shortcut_detected
        metrics = {
            "dp_gap": self.dp_gap_,
            "overall_positive_rate": self.overall_positive_rate_,
        }
        metadata = {
            "min_group_size": self.min_group_size,
            "dp_gap_threshold": self.dp_gap_threshold,
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
