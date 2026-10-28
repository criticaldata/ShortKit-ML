"""Builder for GCE (Generalized Cross Entropy) detector."""

import dataclasses
import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector, resolve_target
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
            attribute_test = None
            _, target_kind = resolve_target(labels, group_labels)
            if self.kwargs.get("gce_attribute_test", True) and target_kind == "attribute":
                # GCE trains on the task label. Its high-loss samples only point
                # to a shortcut on A if they are enriched for some A group.
                attribute_test = _minority_attribute_enrichment(
                    detector.is_minority_, np.asarray(group_labels)
                )
                if attribute_test["p_value"] >= self.kwargs.get("gce_attribute_alpha", 0.05):
                    report = dataclasses.replace(report, risk_level="low")
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
                "attribute_test": attribute_test,
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


def _minority_attribute_enrichment(is_minority: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Chi-square test of GCE minority membership against the attribute."""
    from scipy.stats import chi2_contingency

    is_minority = np.asarray(is_minority, dtype=bool)
    cats = np.unique(groups)
    table = np.array([[np.sum((groups == c) & is_minority), np.sum((groups == c) & ~is_minority)] for c in cats])
    if table.shape[0] < 2 or table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        return {"chi2": 0.0, "p_value": 1.0, "table": table.tolist()}
    chi2, p, _, _ = chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p), "table": table.tolist()}
