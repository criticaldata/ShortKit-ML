"""Builder for intersectional fairness detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import IntersectionalDetector


class IntersectionalDetectorBuilder(BaseDetector):
    def build(self):
        fairness_params = {
            "min_group_size": self.kwargs.get("intersectional_min_group_size", 10),
            "tpr_gap_threshold": self.kwargs.get("intersectional_tpr_gap_threshold", 0.1),
            "fpr_gap_threshold": self.kwargs.get("intersectional_fpr_gap_threshold", 0.1),
            "dp_gap_threshold": self.kwargs.get("intersectional_dp_gap_threshold", 0.1),
            "intersection_attributes": self.kwargs.get("intersection_attributes"),
        }
        estimator = self.kwargs.get("intersectional_estimator")
        return IntersectionalDetector(estimator=estimator, **fairness_params)

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
        print("Running intersectional fairness analysis...")
        if extra_labels is None:
            warnings.warn(
                "Intersectional analysis skipped (extra_labels with 2+ demographics required).",
                stacklevel=2,
            )
            return {
                "success": False,
                "error": "extra_labels with at least 2 demographic attributes (e.g., race, gender) required for intersectional analysis",
            }

        candidate_keys = [
            k for k in extra_labels.keys() if k not in ("spurious", "early_epoch_reps")
        ]
        if len(candidate_keys) < 2:
            warnings.warn(
                "Intersectional analysis skipped: need at least 2 demographic attributes in extra_labels.",
                stacklevel=2,
            )
            return {
                "success": False,
                "error": "Need at least 2 demographic attributes in extra_labels (e.g., {'race': ..., 'gender': ...}). Reserved keys 'spurious', 'early_epoch_reps' are excluded.",
            }

        detector = self.build()
        try:
            detector.fit(embeddings, labels, extra_labels)
            report = detector.report_
            summary_lines = [
                (
                    f"TPR gap: {report.tpr_gap:.3f} | FPR gap: {report.fpr_gap:.3f} | "
                    f"DP gap: {report.dp_gap:.3f} (risk: {str(report.risk_level).upper()})"
                ),
                f"Attributes: {', '.join(report.attribute_names)}",
                f"Reference: {report.reference}",
            ]
            for group, metrics in report.intersection_metrics.items():
                tpr = "nan" if np.isnan(metrics["tpr"]) else f"{metrics['tpr']:.3f}"
                fpr = "nan" if np.isnan(metrics["fpr"]) else f"{metrics['fpr']:.3f}"
                pr = (
                    "nan"
                    if np.isnan(metrics.get("positive_rate", float("nan")))
                    else f"{metrics['positive_rate']:.3f}"
                )
                summary_lines.append(
                    f"  {group}: TPR={tpr} FPR={fpr} pos_rate={pr} (support={metrics['support']:.0f})"
                )
            risk_indicators = []
            if report.risk_level == "high":
                risk_indicators.append("Large disparities across demographic intersections")
            elif report.risk_level == "moderate":
                risk_indicators.append("Moderate disparity in intersectional fairness")
            return {
                "detector": detector,
                "report": report,
                "summary_title": "Fairness (Intersectional)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"Intersectional analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
