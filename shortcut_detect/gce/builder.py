"""Builder for GCE (Generalized Cross Entropy) detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector
from . import GCEDetector


class GCEDetectorBuilder(BaseDetector):
    def build(self):
        return GCEDetector(
            q=self.kwargs.get("gce_q", 0.7),
            loss_percentile_threshold=self.kwargs.get("gce_loss_percentile_threshold", 90.0),
            max_iter=self.kwargs.get("gce_max_iter", 500),
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
        print("Running GCE bias detection...")
        detector = self.build()
        try:
            detector.fit(embeddings, labels)
            report = detector.report_
            summary_lines = [
                f"Minority/bias-conflicting samples: {report.n_minority} ({report.minority_ratio:.1%})",
                f"Loss (mean ± std): {report.loss_mean:.4f} ± {report.loss_std:.4f}",
                f"Loss range: [{report.loss_min:.4f}, {report.loss_max:.4f}]",
                f"Threshold (percentile): {report.threshold:.4f} (q={report.q})",
            ]
            risk_indicators = []
            if report.risk_level == "high":
                risk_indicators.append(
                    f"GCE flagged {report.n_minority} high-loss samples as minority/bias-conflicting"
                )
            elif report.risk_level == "moderate":
                risk_indicators.append(
                    f"GCE flagged {report.n_minority} high-loss samples (moderate shortcut risk)"
                )
            return {
                "detector": detector,
                "report": report,
                "per_sample_losses": detector.per_sample_losses_,
                "is_minority": detector.is_minority_,
                "minority_indices": detector.get_minority_indices(),
                "summary_title": "GCE (Generalized Cross Entropy) Bias Detection",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"GCE detection failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
