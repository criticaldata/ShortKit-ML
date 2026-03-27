"""AIF360 / Fairlearn comparison module.

Runs equivalent fairness analyses using external toolkits (AIF360, Fairlearn)
on the same data and produces comparison tables.

Usage:
    from shortcut_detect.benchmark.baseline_comparison import BaselineComparison

    comp = BaselineComparison()
    results = comp.run(embeddings, labels, group_labels)
    print(results.comparison_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ToolkitResult:
    """Result from running a single toolkit."""

    toolkit_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    supported_features: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Aggregated comparison across toolkits."""

    toolkit_results: dict[str, ToolkitResult] = field(default_factory=dict)

    def comparison_table(self) -> pd.DataFrame:
        """Feature comparison table: rows=features, cols=toolkits.

        Shows which features each toolkit supports based on the actual run.
        """
        all_features: set[str] = set()
        for tr in self.toolkit_results.values():
            all_features.update(tr.supported_features)
        all_features_sorted = sorted(all_features)
        toolkit_names = sorted(self.toolkit_results.keys())

        data: dict[str, list[str]] = {}
        for name in toolkit_names:
            tr = self.toolkit_results[name]
            feature_set = set(tr.supported_features)
            data[name] = ["Yes" if f in feature_set else "No" for f in all_features_sorted]

        return pd.DataFrame(data, index=all_features_sorted)

    def metrics_table(self) -> pd.DataFrame:
        """Metrics comparison table: rows=metrics, cols=toolkits."""
        all_metrics: set[str] = set()
        for tr in self.toolkit_results.values():
            all_metrics.update(tr.metrics.keys())
        all_metrics_sorted = sorted(all_metrics)
        toolkit_names = sorted(self.toolkit_results.keys())

        data: dict[str, list[Any]] = {}
        for name in toolkit_names:
            tr = self.toolkit_results[name]
            data[name] = [tr.metrics.get(m, np.nan) for m in all_metrics_sorted]

        return pd.DataFrame(data, index=all_metrics_sorted)

    def to_latex(self) -> str:
        """LaTeX formatted feature comparison table."""
        table = self.comparison_table()
        return table.to_latex()

    def to_markdown(self) -> str:
        """Markdown formatted feature comparison table."""
        table = self.comparison_table()
        return table.to_markdown()

    def summary(self) -> str:
        """Human-readable summary of comparison results."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("BASELINE COMPARISON SUMMARY")
        lines.append("=" * 60)

        for name in sorted(self.toolkit_results.keys()):
            tr = self.toolkit_results[name]
            lines.append("")
            lines.append(f"--- {tr.toolkit_name} ---")
            if tr.errors:
                lines.append(f"  Errors: {'; '.join(tr.errors)}")
            if tr.metrics:
                for k, v in sorted(tr.metrics.items()):
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.4f}")
                    else:
                        lines.append(f"  {k}: {v}")
            else:
                lines.append("  No metrics computed.")
            lines.append(f"  Features: {', '.join(tr.supported_features) or 'none'}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


def generate_feature_comparison_table() -> pd.DataFrame:
    """Static feature comparison table (no data needed).

    Returns a DataFrame comparing ShortKit-ML, AIF360, and Fairlearn
    across a curated set of capabilities.
    """
    features = {
        "Embedding-space analysis": {"ShortKit-ML": "Yes", "AIF360": "No", "Fairlearn": "No"},
        "Multi-method convergence": {"ShortKit-ML": "Yes", "AIF360": "No", "Fairlearn": "No"},
        "Equalized odds": {"ShortKit-ML": "Yes", "AIF360": "Yes", "Fairlearn": "Yes"},
        "Demographic parity": {"ShortKit-ML": "Yes", "AIF360": "Yes", "Fairlearn": "Yes"},
        "Disparate impact": {"ShortKit-ML": "Limited", "AIF360": "Yes", "Fairlearn": "Yes"},
        "Medical imaging support": {"ShortKit-ML": "Yes", "AIF360": "No", "Fairlearn": "No"},
        "Report generation": {"ShortKit-ML": "Yes", "AIF360": "No", "Fairlearn": "Limited"},
        "Intersectional analysis": {
            "ShortKit-ML": "Yes",
            "AIF360": "Yes",
            "Fairlearn": "Limited",
        },
        "Causal analysis": {"ShortKit-ML": "Yes", "AIF360": "No", "Fairlearn": "No"},
        "Active maintenance": {"ShortKit-ML": "Yes", "AIF360": "Yes", "Fairlearn": "Yes"},
        "Python version": {"ShortKit-ML": "3.8+", "AIF360": "3.8+", "Fairlearn": "3.8+"},
    }
    return pd.DataFrame(features).T


class BaselineComparison:
    """Run fairness analyses across ShortKit-ML, Fairlearn, and AIF360."""

    def __init__(
        self,
        include_fairlearn: bool = True,
        include_aif360: bool = True,
    ) -> None:
        self.include_fairlearn = include_fairlearn
        self.include_aif360 = include_aif360

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        task_labels: np.ndarray | None = None,
    ) -> ComparisonResult:
        """Run comparison across all enabled toolkits.

        Parameters
        ----------
        embeddings : ndarray of shape (n_samples, n_features)
        labels : ndarray of shape (n_samples,)
        group_labels : ndarray of shape (n_samples,)
        task_labels : ndarray of shape (n_samples,), optional
            Alias for labels; kept for API compatibility.

        Returns
        -------
        ComparisonResult
        """
        embeddings = np.asarray(embeddings)
        labels = np.asarray(labels)
        group_labels = np.asarray(group_labels)

        result = ComparisonResult()

        # Always run ShortKit-ML
        result.toolkit_results["ShortKit-ML"] = self._run_shortcutdetect(
            embeddings, labels, group_labels
        )

        if self.include_fairlearn:
            result.toolkit_results["Fairlearn"] = self._run_fairlearn(
                embeddings, labels, group_labels
            )

        if self.include_aif360:
            result.toolkit_results["AIF360"] = self._run_aif360(embeddings, labels, group_labels)

        return result

    # ------------------------------------------------------------------
    # Internal runners
    # ------------------------------------------------------------------

    def _run_shortcutdetect(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> ToolkitResult:
        """Run ShortKit-ML analysis."""
        tr = ToolkitResult(
            toolkit_name="ShortKit-ML",
            supported_features=[
                "embedding_space_analysis",
                "multi_method_convergence",
                "equalized_odds",
                "demographic_parity",
                "report_generation",
            ],
        )
        try:
            from shortcut_detect.unified import ShortcutDetector

            methods = ["probe", "statistical", "equalized_odds", "demographic_parity"]
            detector = ShortcutDetector(methods=methods, seed=42)
            detector.fit(embeddings, labels, group_labels=group_labels)
            results = detector.get_results()

            # Extract key metrics from each method
            for method in methods:
                method_result = results.get(method, {})
                if not method_result.get("success"):
                    continue
                if method == "probe":
                    acc = method_result.get("accuracy")
                    if acc is not None:
                        tr.metrics["probe_accuracy"] = float(acc)
                elif method == "statistical":
                    n_sig = method_result.get("n_significant")
                    if n_sig is not None:
                        tr.metrics["statistical_n_significant"] = int(n_sig)
                elif method == "equalized_odds":
                    gap = method_result.get("max_gap")
                    if gap is not None:
                        tr.metrics["equalized_odds_gap"] = float(gap)
                elif method == "demographic_parity":
                    gap = method_result.get("max_gap")
                    if gap is not None:
                        tr.metrics["demographic_parity_gap"] = float(gap)

            tr.metrics["shortcut_detected"] = any(
                r.get("success") and r.get("shortcut_detected", False)
                for r in results.values()
                if isinstance(r, dict)
            )
        except Exception as exc:
            tr.errors.append(str(exc))
        return tr

    def _run_fairlearn(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> ToolkitResult:
        """Run Fairlearn analysis (optional dependency)."""
        tr = ToolkitResult(
            toolkit_name="Fairlearn",
            supported_features=[
                "equalized_odds",
                "demographic_parity",
            ],
        )
        try:
            from fairlearn.metrics import (
                MetricFrame,
                false_positive_rate,
                selection_rate,
                true_positive_rate,
            )
        except ImportError:
            tr.errors.append("fairlearn is not installed")
            return tr

        try:
            y_pred = self._train_predict(embeddings, labels)

            mf = MetricFrame(
                metrics={
                    "selection_rate": selection_rate,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                },
                y_true=labels,
                y_pred=y_pred,
                sensitive_features=group_labels,
            )

            diff = mf.difference()
            tr.metrics["selection_rate_diff"] = float(diff["selection_rate"])
            tr.metrics["tpr_diff"] = float(diff["true_positive_rate"])
            tr.metrics["fpr_diff"] = float(diff["false_positive_rate"])

            by_group = mf.by_group
            tr.metrics["selection_rate_by_group"] = by_group["selection_rate"].to_dict()
            tr.metrics["tpr_by_group"] = by_group["true_positive_rate"].to_dict()
            tr.metrics["fpr_by_group"] = by_group["false_positive_rate"].to_dict()
        except Exception as exc:
            tr.errors.append(str(exc))

        return tr

    def _run_aif360(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> ToolkitResult:
        """Run AIF360 analysis (optional dependency)."""
        tr = ToolkitResult(
            toolkit_name="AIF360",
            supported_features=[
                "equalized_odds",
                "demographic_parity",
                "disparate_impact",
            ],
        )
        try:
            from aif360.datasets import BinaryLabelDataset  # noqa: F401
            from aif360.metrics import ClassificationMetric  # noqa: F401
        except ImportError:
            tr.errors.append("aif360 is not installed")
            return tr

        try:
            from aif360.datasets import BinaryLabelDataset
            from aif360.metrics import ClassificationMetric

            y_pred = self._train_predict(embeddings, labels)

            g_codes = pd.Categorical(pd.Series(group_labels).astype(str))
            code_arr = g_codes.codes.astype(int)
            valid_mask = code_arr >= 0
            code_values = sorted(np.unique(code_arr[valid_mask]).tolist())

            if len(code_values) < 2:
                tr.errors.append("Need at least two groups for AIF360 analysis")
                return tr

            privileged = int(code_values[0])
            df_eval = pd.DataFrame(
                {
                    "label": labels.astype(int),
                    "pred": np.asarray(y_pred).astype(int),
                    "group": code_arr,
                }
            )
            bld_true = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_eval[["label", "group"]],
                label_names=["label"],
                protected_attribute_names=["group"],
            )
            bld_pred = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_eval[["pred", "group"]].rename(columns={"pred": "label"}),
                label_names=["label"],
                protected_attribute_names=["group"],
            )
            metric = ClassificationMetric(
                bld_true,
                bld_pred,
                privileged_groups=[{"group": privileged}],
                unprivileged_groups=[{"group": int(x)} for x in code_values[1:]],
            )
            tr.metrics["disparate_impact"] = float(metric.disparate_impact())
            tr.metrics["equal_opportunity_difference"] = float(
                metric.equal_opportunity_difference()
            )
            tr.metrics["statistical_parity_difference"] = float(
                metric.statistical_parity_difference()
            )
            tr.metrics["average_abs_odds_difference"] = float(metric.average_abs_odds_difference())
        except Exception as exc:
            tr.errors.append(str(exc))

        return tr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _train_predict(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Train a simple LogisticRegression and return predictions on full data.

        For comparison purposes we train/predict on the same data to keep the
        interface simple; external toolkits only need predictions to compute
        group-level metrics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
            ]
        )
        pipe.fit(embeddings, labels)
        return pipe.predict(embeddings)
