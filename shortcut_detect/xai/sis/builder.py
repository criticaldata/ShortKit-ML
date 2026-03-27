"""Builder for SIS (Sufficient Input Subsets) detector."""

import warnings
from typing import Any

import numpy as np

from ...base_builder import BaseDetector
from .src.detector import SISDetector


class SISDetectorBuilder(BaseDetector):
    def build(self):
        return SISDetector(
            mask_value=self.kwargs.get("sis_mask_value", 0.0),
            max_samples=self.kwargs.get("sis_max_samples", 200),
            test_size=self.kwargs.get("sis_test_size", 0.2),
            shortcut_threshold=self.kwargs.get("sis_shortcut_threshold", 0.15),
            seed=self.seed,
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
        print("Running SIS (Sufficient Input Subsets) detection...")
        detector = self.build()
        try:
            detector.fit(embeddings, labels, group_labels=group_labels)
            report_full = detector.get_report()
            metrics = report_full.get("metrics", {})
            mean_sis = metrics.get("mean_sis_size")
            median_sis = metrics.get("median_sis_size")
            frac_dim = metrics.get("frac_dimensions")
            n_computed = metrics.get("n_computed", 0)

            summary_lines = [
                f"Mean SIS size: {mean_sis:.1f}" if mean_sis is not None else "No SIS computed",
                f"Median SIS size: {median_sis}" if median_sis is not None else "",
                f"Fraction of dimensions: {frac_dim:.1%}" if frac_dim is not None else "",
                f"Samples computed: {n_computed}",
            ]
            summary_lines = [s for s in summary_lines if s]

            risk_indicators = []
            shortcut = report_full.get("shortcut_detected")
            if shortcut is True:
                risk_indicators.append(
                    "Small mean SIS indicates model may rely on few dimensions (potential shortcut)"
                )
            elif shortcut is False:
                risk_indicators.append(
                    "Larger SIS indicates model uses more dimensions (weaker shortcut signal)"
                )

            return {
                "detector": detector,
                "report": report_full.get("report"),
                "metrics": metrics,
                "risk_level": report_full.get("risk_level"),
                "shortcut_detected": shortcut,
                "summary_title": "SIS (Sufficient Input Subsets)",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"SIS detection failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
