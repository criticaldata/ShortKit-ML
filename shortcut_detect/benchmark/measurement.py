"""Precision/Recall measurement harness for shortcut detection evaluation.

Runs all detection methods against synthetic data with known ground truth
shortcut dimensions and computes precision, recall, F1, and Jaccard scores.

Usage:
    from shortcut_detect.benchmark.measurement import MeasurementHarness

    harness = MeasurementHarness(methods=["hbac", "probe", "statistical", "geometric"])
    results = harness.evaluate(embeddings, labels, group_labels, true_shortcut_dims, seed=42)
    # results is a HarnessResult with precision, recall, f1, jaccard per method
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from shortcut_detect.benchmark.method_utils import ALL_METHODS
from shortcut_detect.unified import ShortcutDetector


def probe_permutation_pvalue(
    embeddings: np.ndarray,
    group_labels: np.ndarray,
    n_permutations: int = 100,
    seed: int = 42,
) -> dict:
    """Compute p-value for probe accuracy via permutation null distribution.

    Trains a logistic regression probe on real labels, then on *n_permutations*
    shuffled labels.  Returns a dictionary with observed accuracy, null
    distribution statistics, and the empirical p-value.

    Parameters
    ----------
    embeddings : ndarray of shape ``(n, d)``
    group_labels : ndarray of shape ``(n,)``
    n_permutations : int
        Number of label-shuffled iterations for the null distribution.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``observed_accuracy``, ``null_mean``, ``null_std``, ``p_value``,
        ``n_permutations``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.RandomState(seed)
    X = np.asarray(embeddings, dtype=float)
    y = np.asarray(group_labels).ravel()

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    observed_accuracy = float(np.mean(cross_val_score(clf, X, y, cv=3, scoring="accuracy")))

    null_accuracies = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_accuracies[i] = float(
            np.mean(cross_val_score(clf, X, y_perm, cv=3, scoring="accuracy"))
        )

    p_value = float(np.mean(null_accuracies >= observed_accuracy))

    return {
        "observed_accuracy": observed_accuracy,
        "null_mean": float(np.mean(null_accuracies)),
        "null_std": float(np.std(null_accuracies)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array-like of float
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (default 0.05 for 95 % CI).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``mean``, ``ci_lower``, ``ci_upper``, ``std``, ``n_bootstrap``.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "std": float("nan"),
            "n_bootstrap": n_bootstrap,
        }
    if vals.size == 1:
        v = float(vals[0])
        return {
            "mean": v,
            "ci_lower": v,
            "ci_upper": v,
            "std": 0.0,
            "n_bootstrap": n_bootstrap,
        }

    rng = np.random.RandomState(seed)
    stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = vals[rng.randint(0, vals.size, size=vals.size)]
        stats[i] = float(np.mean(sample))

    ci_lower = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": float(np.mean(vals)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(vals)),
        "n_bootstrap": n_bootstrap,
    }


@dataclass
class MethodResult:
    """Result for a single detection method on one dataset."""

    method: str
    detected: bool
    precision: float
    recall: float
    f1: float
    jaccard: float
    risk_level: str
    raw_result: dict[str, Any]


@dataclass
class HarnessResult:
    """Aggregated result across all methods for one dataset."""

    method_results: list[MethodResult]
    convergence_level: str  # e.g. "3/4"
    convergence_bucket: str  # e.g. "high_confidence"


def method_detected(method: str, result: dict[str, Any]) -> bool:
    """Determine whether a method flagged a shortcut.

    Delegates to the canonical implementation in :mod:`method_utils`.
    """
    from shortcut_detect.benchmark.method_utils import method_flag

    return method_flag(method, result)


def precision_recall_f1(
    predicted_dims: np.ndarray | list[int],
    true_dims: np.ndarray | list[int],
) -> tuple[float, float, float, float]:
    """Compute precision, recall, F1, and Jaccard for dimension-level predictions.

    Parameters
    ----------
    predicted_dims : array-like of int
        Indices of dimensions predicted as shortcut dimensions.
    true_dims : array-like of int
        Indices of the ground-truth shortcut dimensions.

    Returns
    -------
    tuple of (precision, recall, f1, jaccard)
    """
    pred_set = {int(x) for x in np.asarray(predicted_dims).tolist()}
    true_set = {int(x) for x in np.asarray(true_dims).tolist()}
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    jaccard = float(tp / len(pred_set | true_set)) if (pred_set or true_set) else 1.0
    return precision, recall, f1, jaccard


# ---------------------------------------------------------------------------
# Dimension-level score extractors
# ---------------------------------------------------------------------------


def _top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-*k* dimensions by score."""
    if scores.size == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, scores.size))
    order = np.argsort(scores)[::-1]
    return order[:k]


def _probe_dim_scores(X: np.ndarray, group_labels: np.ndarray, seed: int) -> np.ndarray:
    """Compute per-dimension importance via a logistic regression probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed),
            ),
        ]
    )
    model.fit(X, group_labels)
    clf = model.named_steps["clf"]
    coef = np.asarray(clf.coef_, dtype=float)
    if coef.ndim == 2:
        coef = np.mean(np.abs(coef), axis=0)
    return np.asarray(coef, dtype=float)


def _hbac_dim_scores(result: dict[str, Any], embedding_dim: int) -> np.ndarray:
    """Extract per-dimension F-scores from an HBAC result."""
    scores = np.zeros(embedding_dim, dtype=float)
    report = result.get("report", {})
    dim_importance = report.get("dimension_importance")
    if not isinstance(dim_importance, pd.DataFrame):
        return scores
    if "dimension" not in dim_importance.columns or "f_score" not in dim_importance.columns:
        return scores
    for _, row in dim_importance.iterrows():
        name = str(row["dimension"])
        if name.startswith("dim_"):
            try:
                idx = int(name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if 0 <= idx < embedding_dim:
                scores[idx] = float(row.get("f_score", 0.0))
    return scores


def _statistical_dim_scores(result: dict[str, Any], embedding_dim: int) -> np.ndarray:
    """Derive per-dimension scores from corrected p-values."""
    corrected = result.get("corrected_pvalues", {})
    if not isinstance(corrected, dict) or not corrected:
        return np.zeros(embedding_dim, dtype=float)
    stack = []
    for arr in corrected.values():
        vals = np.asarray(arr, dtype=float)
        if vals.shape[0] == embedding_dim:
            stack.append(vals)
    if not stack:
        return np.zeros(embedding_dim, dtype=float)
    min_p = np.min(np.vstack(stack), axis=0)
    return -np.log10(np.clip(min_p, 1e-300, 1.0))


def _geometric_dim_scores(X: np.ndarray, group_labels: np.ndarray) -> np.ndarray:
    """Compute per-dimension centroid-difference scores."""
    groups = np.unique(group_labels)
    if groups.size < 2:
        return np.zeros(X.shape[1], dtype=float)
    centroids = [np.mean(X[group_labels == g], axis=0) for g in groups]
    centroids_arr = np.vstack(centroids)
    scores = np.zeros(X.shape[1], dtype=float)
    n_pairs = 0
    for i in range(centroids_arr.shape[0]):
        for j in range(i + 1, centroids_arr.shape[0]):
            scores += np.abs(centroids_arr[i] - centroids_arr[j])
            n_pairs += 1
    if n_pairs > 0:
        scores /= float(n_pairs)
    return scores


def _risk_level_for(method: str, result: dict[str, Any]) -> str:
    """Return a human-readable risk level string for a method result."""
    if not result.get("success", False):
        return "error"
    if method == "geometric":
        return str(result.get("summary", {}).get("risk_level", "low")).lower()
    if method_detected(method, result):
        return "detected"
    return "not_detected"


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


class MeasurementHarness:
    """Runs detection methods against data and computes precision/recall metrics.

    Parameters
    ----------
    methods : list of str
        Detection methods to evaluate.
    seed : int
        Default random seed.
    """

    SUPPORTED_METHODS = ALL_METHODS

    def __init__(
        self,
        methods: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        if methods is None:
            methods = list(self.SUPPORTED_METHODS)
        for m in methods:
            if m not in self.SUPPORTED_METHODS:
                raise ValueError(f"Unsupported method: {m}")
        self.methods = methods
        self.seed = seed

    # ---- internal helpers ------------------------------------------------

    def _run_methods(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_labels: np.ndarray,
        seed: int,
    ) -> dict[str, dict[str, Any]]:
        """Run each method via :class:`ShortcutDetector` and collect raw results."""
        out: dict[str, dict[str, Any]] = {}
        for method in self.methods:
            detector = ShortcutDetector(
                methods=[method],
                seed=seed,
                statistical_correction="fdr_bh",
                statistical_alpha=0.05,
            )
            try:
                detector.fit(X, y, group_labels=group_labels)
                out[method] = detector.results_.get(
                    method, {"success": False, "error": "missing result"}
                )
            except Exception as exc:
                out[method] = {"success": False, "error": str(exc)}
        return out

    def _dim_scores(
        self,
        method: str,
        X: np.ndarray,
        group_labels: np.ndarray,
        result: dict[str, Any],
        seed: int,
    ) -> np.ndarray:
        """Compute dimension-level scores for a given method."""
        embedding_dim = X.shape[1]
        if method == "probe":
            return _probe_dim_scores(X, group_labels, seed)
        if method == "hbac":
            return _hbac_dim_scores(result, embedding_dim)
        if method == "statistical":
            return _statistical_dim_scores(result, embedding_dim)
        if method == "geometric":
            return _geometric_dim_scores(X, group_labels)
        if method == "bias_direction_pca":
            from shortcut_detect.benchmark.method_utils import bias_direction_pca_dim_scores

            return bias_direction_pca_dim_scores(result, embedding_dim)
        if method == "sis":
            from shortcut_detect.benchmark.method_utils import sis_dim_scores

            return sis_dim_scores(result, embedding_dim)
        # Methods without dim scores
        from shortcut_detect.benchmark.method_utils import nan_dim_scores

        return nan_dim_scores(embedding_dim)

    @staticmethod
    def _convergence_bucket(n_flagged: int, n_methods: int) -> str:
        """Map the number of agreeing methods to a confidence bucket."""
        if n_flagged == n_methods:
            return "high_confidence"
        if n_flagged >= n_methods - 1 and n_methods > 1:
            return "moderate_confidence"
        if n_flagged == 1:
            return "likely_false_alarm"
        if n_flagged == 0:
            return "no_detection"
        return "intermediate"

    # ---- public API ------------------------------------------------------

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        true_shortcut_dims: np.ndarray | list[int],
        seed: int | None = None,
    ) -> HarnessResult:
        """Evaluate all methods on a single dataset.

        Parameters
        ----------
        embeddings : ndarray of shape ``(n, d)``
        labels : ndarray of shape ``(n,)``
        group_labels : ndarray of shape ``(n,)``
        true_shortcut_dims : array-like of int
            Ground-truth shortcut dimension indices.
        seed : int, optional
            Random seed override. Defaults to ``self.seed``.

        Returns
        -------
        HarnessResult
        """
        seed = seed if seed is not None else self.seed
        true_shortcut_dims = np.asarray(true_shortcut_dims, dtype=int)
        k = len(true_shortcut_dims)

        raw_results = self._run_methods(embeddings, labels, group_labels, seed)

        method_results: list[MethodResult] = []
        n_flagged = 0
        for method in self.methods:
            result = raw_results[method]
            detected = method_detected(method, result)
            if detected:
                n_flagged += 1
            scores = self._dim_scores(method, embeddings, group_labels, result, seed)
            pred_dims = _top_k_from_scores(scores, k) if k > 0 else np.array([], dtype=int)
            prec, rec, f1, jacc = precision_recall_f1(pred_dims, true_shortcut_dims)
            method_results.append(
                MethodResult(
                    method=method,
                    detected=detected,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                    jaccard=jacc,
                    risk_level=_risk_level_for(method, result),
                    raw_result=result,
                )
            )

        convergence_level = f"{n_flagged}/{len(self.methods)}"
        convergence_bucket = self._convergence_bucket(n_flagged, len(self.methods))

        return HarnessResult(
            method_results=method_results,
            convergence_level=convergence_level,
            convergence_bucket=convergence_bucket,
        )

    def evaluate_batch(
        self,
        datasets: list[dict],
        seeds: list[int] | None = None,
    ) -> pd.DataFrame:
        """Run :meth:`evaluate` on multiple datasets and return a results table.

        Parameters
        ----------
        datasets : list of dict
            Each dict must contain keys ``"embeddings"``, ``"labels"``,
            ``"group_labels"``, and ``"true_shortcut_dims"``.
        seeds : list of int, optional
            Per-dataset seeds.  If ``None``, uses ``self.seed`` for all.

        Returns
        -------
        pd.DataFrame
            One row per (dataset_index, method) plus a convergence row.
        """
        if seeds is None:
            seeds = [self.seed] * len(datasets)
        if len(seeds) != len(datasets):
            raise ValueError("len(seeds) must match len(datasets)")

        rows: list[dict[str, Any]] = []
        for idx, (ds, seed) in enumerate(zip(datasets, seeds, strict=False)):
            hr = self.evaluate(
                embeddings=ds["embeddings"],
                labels=ds["labels"],
                group_labels=ds["group_labels"],
                true_shortcut_dims=ds["true_shortcut_dims"],
                seed=seed,
            )
            for mr in hr.method_results:
                rows.append(
                    {
                        "dataset_index": idx,
                        "seed": seed,
                        "method": mr.method,
                        "detected": mr.detected,
                        "precision": mr.precision,
                        "recall": mr.recall,
                        "f1": mr.f1,
                        "jaccard": mr.jaccard,
                        "risk_level": mr.risk_level,
                        "convergence_level": hr.convergence_level,
                        "convergence_bucket": hr.convergence_bucket,
                    }
                )
        return pd.DataFrame(rows)
