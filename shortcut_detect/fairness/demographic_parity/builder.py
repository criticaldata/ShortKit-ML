"""Builder for demographic parity detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import DemographicParityDetector


class DemographicParityDetectorBuilder(BaseDetector):
    def build(self):
        fairness_params = {
            "min_group_size": self.kwargs.get("dp_min_group_size", 10),
            "dp_gap_threshold": self.kwargs.get("dp_gap_threshold", 0.1),
        }
        estimator = self.kwargs.get("dp_estimator")
        return DemographicParityDetector(estimator=estimator, **fairness_params)

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        feature_names: list[str] | None,
        protected_labels: np.ndarray | None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        print("Running demographic parity fairness analysis...")
        if protected_labels is None:
            warnings.warn(
                "Demographic parity analysis skipped (group_labels are required).", stacklevel=2
            )
            return {
                "success": False,
                "error": "group_labels required for demographic parity analysis",
            }

        detector = self.build()
        try:
            detector.fit(embeddings, labels, protected_labels)
            report = detector.report_
            summary_lines = [
                f"DP gap: {report.dp_gap:.3f} (risk: {report.risk_level.upper()})",
                f"Overall positive rate: {report.overall_positive_rate:.3f}",
            ]
            for group, metrics in report.group_rates.items():
                rate = (
                    "nan"
                    if np.isnan(metrics["positive_rate"])
                    else f"{metrics['positive_rate']:.3f}"
                )
                summary_lines.append(
                    f"  {group}: positive_rate={rate} (support={metrics['support']:.0f})"
                )
            risk_indicators = []
            if report.risk_level == "high":
                risk_indicators.append("Demographic parity gap is HIGH across groups")
            elif report.risk_level == "moderate":
                risk_indicators.append("Demographic parity gap is moderate")
            return {
                "detector": detector,
                "report": report,
                "summary_title": "Fairness (Demographic Parity)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"Demographic parity analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
