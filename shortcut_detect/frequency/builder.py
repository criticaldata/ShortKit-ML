"""Builder for frequency shortcut detector."""

from typing import Any

import numpy as np

from ..base_builder import BaseDetector
from .detector import FrequencyDetector


class FrequencyShortcutBuilder(BaseDetector):
    """Minimal builder wrapper for FrequencyDetector."""

    def build(self) -> FrequencyDetector:
        top_percent = self.kwargs.get("freq_top_percent", 0.05)
        tpr_threshold = self.kwargs.get("freq_tpr_threshold", 0.5)
        fpr_threshold = self.kwargs.get("freq_fpr_threshold", 0.15)
        probe_estimator = self.kwargs.get("freq_probe_estimator", None)
        probe_eval = self.kwargs.get("freq_probe_evaluation", "train")
        probe_holdout_frac = self.kwargs.get("freq_probe_holdout_frac", 0.2)
        random_state = self.kwargs.get("freq_random_state", self.seed)

        return FrequencyDetector(
            top_percent=top_percent,
            tpr_threshold=tpr_threshold,
            fpr_threshold=fpr_threshold,
            probe_estimator=probe_estimator,
            probe_evaluation=probe_eval,
            probe_holdout_frac=probe_holdout_frac,
            random_state=random_state,
        )

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        detector = self.build()
        try:
            detector.fit(embeddings=embeddings, labels=labels)
            report = detector.get_report()
            shortcut_detected = report.get("shortcut_detected", False)
            risk = report.get("risk_level", "unknown").upper()
            summary_lines = [
                f"Detected shortcuts: {'YES' if shortcut_detected else 'NO'}",
                f"Risk level: {risk}",
                f"Classes with shortcuts: {report.get('report', {}).get('shortcut_classes', [])}",
                f"Top-percent used: {report.get('metrics', {}).get('top_percent', self.kwargs.get('freq_top_percent', 0.05))}",
            ]
            risk_indicators = []
            if shortcut_detected:
                risk_indicators.append("Embedding-space sensitivity indicates shortcut signatures")

            return {
                "detector": detector,
                "report": report,
                "summary_title": "Embedding Frequency Shortcut (embedding-only)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}
