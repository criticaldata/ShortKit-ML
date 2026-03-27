"""Builder for causal effect detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import CausalEffectDetector


class CausalEffectDetectorBuilder(BaseDetector):
    def build(self):
        return CausalEffectDetector(
            effect_estimator=self.kwargs.get("causal_effect_estimator", "direct"),
            spurious_threshold=self.kwargs.get("causal_effect_spurious_threshold", 0.1),
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
        raise ValueError(
            "causal_effect requires attributes. Use ShortcutDetector.fit_from_loaders."
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
            raise ValueError("Loader for causal_effect must return a dict.")

        detector = self.build()
        try:
            embeddings = data.get("embeddings")
            labels = data.get("labels")
            attributes = data.get("attributes")
            if embeddings is None or labels is None or attributes is None:
                raise ValueError(
                    "Loader for causal_effect must provide 'embeddings', 'labels', and 'attributes'."
                )
            detector.fit(
                embeddings=embeddings,
                labels=labels,
                attributes=attributes,
                counterfactual_pairs=data.get("counterfactual_pairs"),
            )
            report = detector.get_report()
            metrics = report.get("metrics", {})
            n_spurious = metrics.get("n_spurious", 0)
            effects = metrics.get("per_attribute_effects", {})
            summary_lines = [
                f"Attributes: {metrics.get('n_attributes', 0)} (spurious: {n_spurious})",
                f"Threshold: {metrics.get('spurious_threshold')}",
            ]
            for attr_name, effect in list(effects.items())[:5]:
                summary_lines.append(f"  {attr_name}: causal_effect={effect:.3f}")
            risk_indicators = []
            if n_spurious > 0:
                risk_indicators.append(
                    f"Causal effect analysis flagged {n_spurious} spurious attribute(s)"
                )
            return {
                "detector": detector,
                "report": report.get("report", {}),
                "summary_title": "Causal Effect Regularization",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
                "metrics": metrics,
                "metadata": report.get("metadata", {}),
                "shortcut_detected": report.get("shortcut_detected"),
            }
        except Exception as exc:
            warnings.warn(f"Causal effect analysis failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
