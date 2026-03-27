"""Builder for CAV (Concept Activation Vectors) detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import CAVDetector


class CAVDetectorBuilder(BaseDetector):
    def build(self):
        return CAVDetector(
            classifier=self.kwargs.get("cav_classifier", "logreg"),
            random_state=self.seed,
            test_size=self.kwargs.get("cav_test_size", 0.2),
            min_examples_per_set=self.kwargs.get("cav_min_examples_per_set", 20),
            shortcut_threshold=self.kwargs.get("cav_shortcut_threshold", 0.6),
            quality_threshold=self.kwargs.get("cav_quality_threshold", 0.7),
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
        raise ValueError(
            "cav requires concept/random activation sets. Use ShortcutDetector.fit_from_loaders."
        )

    def run_from_loader(
        self,
        loader: Any,
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        data = loader() if callable(loader) else loader
        if not isinstance(data, dict):
            raise ValueError("Loader for cav must return a dict.")

        detector = self.build()
        try:
            detector.fit(
                concept_sets=data.get("concept_sets"),
                random_set=data.get("random_set"),
                target_activations=data.get("target_activations"),
                target_directional_derivatives=data.get("target_directional_derivatives"),
            )
            report = detector.get_report()
            metrics = report.get("metrics", {})
            per_concept = report.get("report", {}).get("per_concept", [])
            n_flagged = int(sum(1 for row in per_concept if row.get("flagged")))
            summary_lines = [
                f"Concepts: {metrics.get('n_concepts', 0)} (tested: {metrics.get('n_tested', 0)})",
                f"Max TCAV score: {metrics.get('max_tcav_score')}",
                f"Max concept quality (AUC): {metrics.get('max_concept_quality')}",
                f"Flagged concepts: {n_flagged}",
            ]
            risk_indicators = []
            if n_flagged > 0:
                risk_indicators.append(f"CAV flagged {n_flagged} concepts above thresholds")

            return {
                "detector": detector,
                "report": report.get("report", {}),
                "summary_title": "CAV (Concept Activation Vectors)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
                "metrics": metrics,
                "metadata": report.get("metadata", {}),
                "details": report.get("details", {}),
                "shortcut_detected": report.get("shortcut_detected"),
            }
        except Exception as exc:
            warnings.warn(f"CAV analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
