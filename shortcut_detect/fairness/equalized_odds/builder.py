"""Builder for equalized odds detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import EqualizedOddsDetector


class EqualizedOddsDetectorBuilder(BaseDetector):
    def build(self):
        fairness_params = {
            "min_group_size": self.kwargs.get("eo_min_group_size", 10),
            "tpr_gap_threshold": self.kwargs.get("eo_tpr_gap_threshold", 0.1),
            "fpr_gap_threshold": self.kwargs.get("eo_fpr_gap_threshold", 0.1),
        }
        estimator = self.kwargs.get("eo_estimator")
        return EqualizedOddsDetector(estimator=estimator, **fairness_params)

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
        print("Running equalized odds fairness analysis...")
        if protected_labels is None:
            warnings.warn(
                "Equalized odds analysis skipped (group_labels are required).", stacklevel=2
            )
            return {
                "success": False,
                "error": "group_labels required for equalized odds analysis",
            }

        detector = self.build()
        try:
            detector.fit(embeddings, labels, protected_labels)
            report = detector.report_
            summary_lines = [
                (
                    f"TPR gap: {report.tpr_gap:.3f} | FPR gap: {report.fpr_gap:.3f} "
                    f"(risk: {report.risk_level.upper()})"
                ),
                f"Reference: {report.reference}",
            ]
            for group, metrics in report.group_metrics.items():
                tpr = "nan" if np.isnan(metrics["tpr"]) else f"{metrics['tpr']:.3f}"
                fpr = "nan" if np.isnan(metrics["fpr"]) else f"{metrics['fpr']:.3f}"
                summary_lines.append(
                    f"  {group}: TPR={tpr} FPR={fpr} (support={metrics['support']:.0f})"
                )
            risk_indicators = []
            if report.risk_level == "high":
                risk_indicators.append("Equalized odds gaps are HIGH across groups")
            elif report.risk_level == "moderate":
                risk_indicators.append("Equalized odds gaps show moderate disparity")
            return {
                "detector": detector,
                "report": report,
                "summary_title": "Fairness (Equalized Odds)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"Equalized odds analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
