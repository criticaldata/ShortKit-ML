"""Builder for statistical detector."""

import warnings
from typing import Any

import numpy as np

from ..base_builder import BaseDetector
from . import GroupDiffTest


class StatisticalDetectorBuilder(BaseDetector):
    def build(self):
        from scipy.stats import mannwhitneyu

        test_fn = self.kwargs.get("statistical_test", mannwhitneyu)
        return GroupDiffTest(test=test_fn)

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
        print("Running statistical tests...")
        stat_test = self.build()
        try:
            stat_test.fit(embeddings, group_labels)
            p_values = stat_test.get_pvalues()
            alpha = self.kwargs.get("statistical_alpha", 0.05)
            correction_method = self.kwargs.get("statistical_correction", "fdr_bh")
            significant_raw = stat_test.apply_threshold(alpha=alpha, verbose=False)
            correction_results = stat_test.apply_correction(
                alpha=alpha, method=correction_method, verbose=False
            )
            significant = correction_results["significant_features"]
            total_comparisons = len(significant)
            comparisons_with_sig = sum(1 for v in significant.values() if v is not None)
            summary_lines = [
                f"Comparisons performed: {total_comparisons}",
                f"Multiple testing correction: {correction_method.upper()} (α={alpha})",
                f"Comparisons with significant features: {comparisons_with_sig}",
            ]
            for comparison, features in significant.items():
                raw_features = significant_raw.get(comparison)
                n_raw = len(raw_features) if raw_features else 0
                n_corrected = len(features) if features else 0
                if n_raw > 0 or n_corrected > 0:
                    summary_lines.append(
                        f"  [{comparison}]: {n_corrected} significant (was {n_raw} before correction)"
                    )
            risk_indicators = []
            if any(v is not None and len(v) > 0 for v in significant.values()):
                n_sig = sum(1 for v in significant.values() if v is not None and len(v) > 0)
                risk_indicators.append(f"{n_sig} group comparisons show significant differences")
            return {
                "detector": stat_test,
                "p_values": p_values,
                "significant_features_raw": significant_raw,
                "significant_features": significant,
                "corrected_pvalues": correction_results["corrected_pvalues"],
                "correction_method": correction_method,
                "alpha": alpha,
                "summary_title": "Statistical Testing",
                "summary_lines": summary_lines,
                "risk_indicators": risk_indicators,
                "success": True,
            }
        except Exception as exc:
            warnings.warn(f"Statistical testing failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
