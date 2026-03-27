"""Builder for geometric shortcut analyzer."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import GeometricShortcutAnalyzer


class GeometricDetectorBuilder(BaseDetector):
    def build(self):
        geom_params = {
            "n_components": self.kwargs.get("geometric_n_components", 5),
            "min_group_size": self.kwargs.get("geometric_min_group_size", 20),
            "effect_threshold": self.kwargs.get("geometric_effect_threshold", 0.8),
            "subspace_cosine_threshold": self.kwargs.get(
                "geometric_subspace_cosine_threshold", 0.85
            ),
        }
        return GeometricShortcutAnalyzer(**geom_params)

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
        print("Running geometric shortcut analysis...")
        geom = self.build()
        try:
            geom.fit(embeddings, group_labels)
            summary = geom.summary_
            summary_lines = [summary["message"]]
            bias_pairs = geom.bias_pairs_[:3]
            if bias_pairs:
                summary_lines.append("Top bias directions:")
                for pair in bias_pairs:
                    summary_lines.append(
                        f"  {pair.groups[0]} vs {pair.groups[1]}: effect_size={pair.effect_size:.2f}, "
                        f"alignment={pair.alignment_score:.2f}"
                    )
            subspace_pairs = geom.subspace_pairs_[:3]
            if subspace_pairs:
                summary_lines.append("Prototype subspace overlap:")
                for pair in subspace_pairs:
                    summary_lines.append(
                        f"  {pair.groups[0]} vs {pair.groups[1]}: mean_cosine={pair.mean_cosine:.2f}, "
                        f"min_angle={pair.min_angle_deg:.1f}°"
                    )
            risk_indicators = []
            if summary.get("risk_level") == "high":
                risk_indicators.append("Geometric analysis flagged HIGH risk")
            elif summary.get("risk_level") == "moderate":
                risk_indicators.append("Geometric analysis signaled moderate shortcut risk")
            return {
                "detector": geom,
                "bias_pairs": geom.bias_pairs_,
                "subspace_pairs": geom.subspace_pairs_,
                "summary": summary,
                "summary_title": "Geometric Bias Analysis",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"Geometric analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
