"""Builder for HBAC detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector
from . import HBACDetector


class HBACDetectorBuilder(BaseDetector):
    def build(self):
        return HBACDetector(
            max_iterations=self.kwargs.get("hbac_max_iterations", 3),
            min_cluster_size=self.kwargs.get("hbac_min_cluster_size", 0.01),
            random_state=self.seed,
        )

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
        print("Running HBAC detection...")
        detector = self.build()
        try:
            detector.fit(embeddings, labels, feature_names=feature_names)
            report = detector.shortcut_report_
            shortcut_info = report["has_shortcut"]
            summary_lines = [
                f"Shortcut detected: {'YES' if shortcut_info['exists'] else 'NO'}",
            ]
            if shortcut_info["types"]:
                summary_lines.append(f"Types: {', '.join(shortcut_info['types'])}")
            summary_lines.append(f"Clusters found: {len(detector.clusters_)}")
            risk_indicators = []
            if shortcut_info["exists"]:
                risk_indicators.append(
                    f"HBAC detected shortcuts ({shortcut_info['confidence']} confidence)"
                )
            return {
                "detector": detector,
                "report": report,
                "summary_title": "HBAC (Clustering-based Detection)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"HBAC detection failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
