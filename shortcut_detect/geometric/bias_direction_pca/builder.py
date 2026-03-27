"""Builder for bias direction PCA detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import BiasDirectionPCADetector


class BiasDirectionPCADetectorBuilder(BaseDetector):
    def build(self):
        params = {
            "n_components": self.kwargs.get("pca_components", 1),
            "min_group_size": self.kwargs.get("pca_min_group_size", 10),
            "gap_threshold": self.kwargs.get("pca_gap_threshold", 0.5),
        }
        return BiasDirectionPCADetector(**params)

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
        print("Running bias direction (PCA) analysis...")
        if protected_labels is None:
            warnings.warn(
                "Bias direction analysis skipped (group_labels are required).", stacklevel=2
            )
            return {
                "success": False,
                "error": "group_labels required for bias direction analysis",
            }

        detector = self.build()
        try:
            detector.fit(embeddings, protected_labels)
            report = detector.report_
            summary_lines = [
                f"Projection gap: {report.projection_gap:.3f} (risk: {report.risk_level.upper()})",
                f"Explained variance: {report.explained_variance:.3f}",
            ]
            for group, metrics in report.group_projections.items():
                summary_lines.append(
                    f"  {group}: projection={metrics['projection']:.3f} "
                    f"(support={metrics['support']:.0f})"
                )
            risk_indicators = []
            if report.risk_level == "high":
                risk_indicators.append("Bias direction gap is HIGH across groups")
            elif report.risk_level == "moderate":
                risk_indicators.append("Bias direction gap is moderate")
            return {
                "detector": detector,
                "report": report,
                "summary_title": "Embedding Bias Direction (PCA)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"Bias direction analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
