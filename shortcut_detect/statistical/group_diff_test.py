import itertools
from typing import Any

import numpy as np

from ..detector_base import DetectorBase


class FeatureGroupDiffTest(DetectorBase):
    """
    Generic wrapper for feature-wise statistical tests that returns only p-values.

    Automatically handles:
      - 2-class problems (single comparison)
      - 3+ classes → performs both one-vs-one and one-vs-rest tests.

    Parameters
    ----------
    test : callable
        Function handle for the test (e.g. scipy.stats.ttest_ind, sklearn.feature_selection.f_classif, or lambda).
        Must return either (statistic, p_value) or just a p_value.
    **test_kwargs : dict
        Extra keyword arguments passed to the test function.
    """

    def __init__(self, test, **test_kwargs):
        super().__init__(method="statistical")
        if not callable(test):
            raise TypeError("`test` must be callable (like a MATLAB function handle).")
        self.test = test
        self.test_kwargs = test_kwargs
        self.p_values_ = None
        self.shortcut_detected_ = None

    def _validate_inputs(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (samples × features).")
        unique_vals = np.unique(Y)
        return X, Y, unique_vals

    def _apply_test(self, X1, X2):
        """Applies the test to two matrices feature-wise and returns p-values."""
        n_features = X1.shape[1]
        test_func = self.test

        p_values = []
        for i in range(n_features):
            res = test_func(X1[:, i], X2[:, i], **self.test_kwargs)
            if isinstance(res, tuple | list) and len(res) >= 2:
                p = res[1]
            else:
                p = res
            p_values.append(p)
        return np.array(p_values)

    def fit(self, X, Y):
        """
        Computes feature-wise p-values for:
          - binary case: one comparison
          - 3+ classes: all one-vs-one and one-vs-rest comparisons.
        """
        X, Y, unique_vals = self._validate_inputs(X, Y)
        results = {}

        # Binary case
        if len(unique_vals) == 2:
            X1, X2 = X[Y == unique_vals[0]], X[Y == unique_vals[1]]
            results[f"{unique_vals[0]}_vs_{unique_vals[1]}"] = self._apply_test(X1, X2)

        # Multi-class case
        elif len(unique_vals) >= 3:
            # --- One-vs-One ---
            for a, b in itertools.combinations(unique_vals, 2):
                X1, X2 = X[Y == a], X[Y == b]
                key = f"{a}_vs_{b}"
                results[key] = self._apply_test(X1, X2)

            # --- One-vs-Rest ---
            for a in unique_vals:
                X1 = X[Y == a]
                X2 = X[Y != a]
                key = f"{a}_vs_rest"
                results[key] = self._apply_test(X1, X2)

        self.p_values_ = results
        self.shortcut_detected_ = None
        metrics = {"n_comparisons": len(results)}
        metadata = {"n_features": X.shape[1], "n_classes": len(unique_vals)}
        self._set_results(
            shortcut_detected=None,
            risk_level="unknown",
            metrics=metrics,
            notes="Run apply_threshold() or apply_correction() to identify significant features.",
            metadata=metadata,
            report={"p_values": results},
        )
        self._is_fitted = True
        return self

    def get_pvalues(
        self,
    ):
        """Return a dictionary mapping comparison names → p-values array."""
        if self.p_values_ is None:
            raise RuntimeError("Run `.fit(X, Y)` before calling `.get_pvalues()`.")
        return self.p_values_

    def apply_threshold(self, alpha=0.05, verbose=True):
        """
        Apply a significance threshold to the p-values and return features
        that meet the threshold (i.e., p < alpha).

        Parameters
        ----------
        alpha : float, optional
            Significance level to threshold p-values. Default is 0.05.
        verbose : bool, optional
            If True, prints a short summary for each comparison.

        Returns
        -------
        dict
            Dictionary mapping comparison names → list of significant feature indices.
        """
        if self.p_values_ is None:
            raise RuntimeError("Run `.fit(X, Y)` before calling `.apply_threshold()`.")
        if verbose:
            print("SUMMARY:")
        significant_features = {}
        for key, pvals in self.p_values_.items():
            sig_idx = np.where(pvals < alpha)[0].tolist()
            if sig_idx:  # if not empty list is found
                significant_features[key] = sig_idx
            else:
                significant_features[key] = None

            if verbose:
                n_features = len(pvals)
                n_significant = len(sig_idx)
                perc_significant = (n_significant / n_features) * 100
                print(
                    f"[{key}] {n_significant}/{n_features} features significant "
                    f"({perc_significant:.1f}% below α={alpha})"
                )

        self._update_results_from_significance(significant_features, alpha, method="threshold")
        return significant_features

    def apply_correction(
        self, alpha: float = 0.05, method: str = "fdr_bh", verbose: bool = True
    ) -> dict:
        """
        Apply multiple testing correction to p-values.

        When testing across many embedding dimensions (e.g., 512 dimensions),
        multiple testing correction is essential to control false positives.
        Without correction, testing 512 dimensions at α=0.05 would yield ~25
        false positives by chance alone.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05)
        method : str
            Correction method:
            - 'bonferroni': Bonferroni correction (FWER control, most conservative)
            - 'holm': Holm step-down (FWER control, less conservative than Bonferroni)
            - 'fdr_bh': Benjamini-Hochberg (FDR control, recommended for most cases)
            - 'fdr_by': Benjamini-Yekutieli (FDR control for dependent tests)
        verbose : bool
            Print summary of results

        Returns
        -------
        dict
            Dictionary with keys:
            - 'significant_features': dict mapping comparison to list of significant dimension indices
            - 'corrected_pvalues': dict mapping comparison to corrected p-value arrays
            - 'rejected': dict mapping comparison to boolean mask arrays
            - 'method': the correction method used
            - 'alpha': the significance level used

        References
        ----------
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate:
        A Practical and Powerful Approach to Multiple Testing. Journal of the Royal
        Statistical Society. Series B, 57(1), 289-300.

        Examples
        --------
        >>> from scipy.stats import mannwhitneyu
        >>> test = FeatureGroupDiffTest(test=mannwhitneyu)
        >>> test.fit(embeddings, labels)
        >>> results = test.apply_correction(method='fdr_bh', alpha=0.05)
        >>> print(f"Significant dimensions: {results['significant_features']}")
        """
        from statsmodels.stats.multitest import multipletests

        if self.p_values_ is None:
            raise RuntimeError("Run `.fit(X, Y)` before calling `.apply_correction()`.")

        valid_methods = ["bonferroni", "holm", "fdr_bh", "fdr_by"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        results = {
            "significant_features": {},
            "corrected_pvalues": {},
            "rejected": {},
            "method": method,
            "alpha": alpha,
        }

        if verbose:
            print(f"Multiple Testing Correction ({method.upper()}, α={alpha}):")
            print("-" * 50)

        for comparison, pvals in self.p_values_.items():
            # Apply correction
            rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)

            # Store results
            results["rejected"][comparison] = rejected
            results["corrected_pvalues"][comparison] = pvals_corrected
            sig_indices = np.where(rejected)[0].tolist()
            results["significant_features"][comparison] = sig_indices if sig_indices else None

            if verbose:
                n_sig = np.sum(rejected)
                n_total = len(pvals)
                n_raw = np.sum(pvals < alpha)
                print(
                    f"  [{comparison}] {n_sig}/{n_total} significant after correction "
                    f"(was {n_raw} before)"
                )

        self._update_results_from_significance(
            results.get("significant_features", {}), alpha, method=method
        )
        return results

    def _update_results_from_significance(
        self,
        significant_features: dict[str, list | None],
        alpha: float,
        method: str,
    ) -> None:
        has_significant = any(v is not None and len(v) > 0 for v in significant_features.values())
        self.shortcut_detected_ = True if has_significant else False
        risk_level = "moderate" if has_significant else "low"
        metrics = {
            "n_comparisons": len(significant_features),
            "comparisons_with_significant": sum(
                1 for v in significant_features.values() if v is not None and len(v) > 0
            ),
            "alpha": alpha,
            "correction_method": method,
        }
        report = {
            "p_values": self.p_values_,
            "significant_features": significant_features,
            "alpha": alpha,
            "correction_method": method,
        }
        self._set_results(
            shortcut_detected=self.shortcut_detected_,
            risk_level=risk_level,
            metrics=metrics,
            notes="Statistical significance indicates potential shortcut dimensions.",
            metadata=self.results_.get("metadata", {}),
            report=report,
        )

    def get_report(self) -> dict[str, Any]:
        return super().get_report()


if __name__ == "__main__":
    from scipy.stats import mannwhitneyu

    # Example usage
    X = np.random.rand(1000000, 5)  # 1,000,000 samples, 5 features
    Y = np.random.choice(["A", "B", "C"], size=1000000)  # 3 classes

    test = FeatureGroupDiffTest(test=mannwhitneyu)
    test.fit(X, Y)
    p_values = test.get_pvalues()

    for comparison, pvals in p_values.items():
        print(f"{comparison}: {pvals}")

    res = test.apply_threshold(alpha=0.05, verbose=True)
    print(res)
