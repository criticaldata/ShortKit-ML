"""Sensitivity sweep modules for shortcut detection benchmarks.

Runs targeted parameter sweeps to evaluate detection robustness:
- B04: Sample size sweep (100, 250, 500, 1000, 5000)
- B05: Class imbalance sweep (50/50, 70/30, 90/10)
- B06: Embedding dimensionality sweep (32, 128, 512, 1024, 2048)

Usage:
    from shortcut_detect.benchmark.sensitivity import SensitivitySweep

    sweep = SensitivitySweep(methods=["hbac", "probe", "statistical", "geometric"])

    # B04: Sample size
    results = sweep.sweep_sample_size(
        sample_sizes=[100, 250, 500, 1000, 5000],
        effect_size=0.8, embedding_dim=128, n_seeds=10
    )

    # B05: Class imbalance
    results = sweep.sweep_imbalance(
        group_ratios=[0.5, 0.7, 0.9],
        effect_size=0.8, n_samples=1000, n_seeds=10
    )

    # B06: Embedding dimensionality
    results = sweep.sweep_dimensionality(
        embedding_dims=[32, 128, 512, 1024, 2048],
        effect_size=0.8, n_samples=1000, n_seeds=10
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shortcut_detect.benchmark.method_utils import ALL_METHODS as SUPPORTED_METHODS
from shortcut_detect.benchmark.method_utils import method_flag as _shared_method_flag
from shortcut_detect.unified import ShortcutDetector

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Container for sensitivity sweep results.

    Attributes:
        sweep_param: Name of the parameter being swept (e.g. ``"sample_size"``).
        param_values: List of values the swept parameter took.
        results_df: Per-run results with columns *param_value*, *method*,
            *seed*, *detected*, *precision*, *recall*, *f1*,
            *convergence_count*.
    """

    sweep_param: str
    param_values: list
    results_df: pd.DataFrame

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Return aggregated mean/std per *param_value* and *method*."""
        numeric_cols = ["detected", "precision", "recall", "f1", "convergence_count"]
        present = [c for c in numeric_cols if c in self.results_df.columns]
        grouped = self.results_df.groupby(["param_value", "method"])[present]
        agg = grouped.agg(["mean", "std"]).reset_index()
        # Flatten multi-level columns
        agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns]
        return agg

    def to_csv(self, path: str | Path) -> None:
        """Persist raw results to *path* as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.results_df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Helper: detect whether a method flagged a shortcut
# ---------------------------------------------------------------------------


def _method_flag(method: str, result: dict[str, Any]) -> bool:
    """Determine if *method* flagged a shortcut in *result*."""
    return _shared_method_flag(method, result)


# ---------------------------------------------------------------------------
# Helper: precision / recall / f1 at the dimension level
# ---------------------------------------------------------------------------


def _precision_recall_f1(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    pred_set = {int(x) for x in pred.tolist()}
    true_set = {int(x) for x in truth.tolist()}
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Helper: dimension-level scoring per method
# ---------------------------------------------------------------------------


def _probe_dim_scores(X: np.ndarray, group_labels: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)),
        ]
    )
    model.fit(X, group_labels)
    clf = model.named_steps["clf"]
    coef = np.asarray(clf.coef_, dtype=float)
    if coef.ndim == 2:
        coef = np.mean(np.abs(coef), axis=0)
    return np.asarray(coef, dtype=float)


def _hbac_dim_scores(result: dict[str, Any], embedding_dim: int) -> np.ndarray:
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


def _top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, scores.size))
    order = np.argsort(scores)[::-1]
    return order[:k]


# ---------------------------------------------------------------------------
# Main sweep class
# ---------------------------------------------------------------------------


class SensitivitySweep:
    """Run targeted sensitivity sweeps over synthetic shortcut datasets.

    Parameters
    ----------
    methods : list[str]
        Detection methods to evaluate. Must be a subset of
        ``("hbac", "probe", "statistical", "geometric")``.
    shortcut_dims : int
        Number of embedding dimensions that carry the shortcut signal.
    base_seed : int
        Base random seed used to derive per-run seeds.
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        shortcut_dims: int = 5,
        base_seed: int = 42,
    ) -> None:
        if methods is None:
            methods = list(SUPPORTED_METHODS)
        for m in methods:
            if m not in SUPPORTED_METHODS:
                raise ValueError(f"Unsupported method '{m}'. Supported: {SUPPORTED_METHODS}")
        self.methods = list(methods)
        self.shortcut_dims = shortcut_dims
        self.base_seed = base_seed

    # ------------------------------------------------------------------
    # Seed generation (mirrors PaperBenchmarkRunner._seed_values)
    # ------------------------------------------------------------------
    def _seed_values(self, count: int, offset: int = 0) -> list[int]:
        rng = np.random.RandomState(self.base_seed + offset)
        values = rng.choice(np.arange(1, 1_000_000, dtype=np.int64), size=count, replace=False)
        return [int(v) for v in values]

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_synthetic_dataset(
        *,
        n_samples: int,
        embedding_dim: int,
        shortcut_dims: int,
        effect_size: float,
        group_ratio: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a synthetic shortcut dataset.

        The generation logic mirrors
        ``PaperBenchmarkRunner._generate_synthetic_dataset``.
        """
        rng = np.random.RandomState(seed)
        y = (rng.rand(n_samples) > group_ratio).astype(np.int64)
        X = rng.randn(n_samples, embedding_dim).astype(np.float32)
        true_dims = np.arange(shortcut_dims, dtype=int)

        if effect_size > 0:
            for dim in true_dims:
                X[y == 0, dim] -= effect_size
                X[y == 1, dim] += effect_size
        return X, y, true_dims

    # ------------------------------------------------------------------
    # Single-run evaluation
    # ------------------------------------------------------------------
    def _evaluate_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        true_dims: np.ndarray,
        embedding_dim: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        """Run all methods on a single dataset and return per-method rows."""
        method_results: dict[str, dict[str, Any]] = {}
        for method in self.methods:
            detector = ShortcutDetector(
                methods=[method],
                seed=seed,
                statistical_correction="fdr_bh",
                statistical_alpha=0.05,
            )
            try:
                detector.fit(X, y, group_labels=y)
                method_results[method] = detector.results_.get(
                    method, {"success": False, "error": "missing result"}
                )
            except Exception:
                method_results[method] = {"success": False, "error": "exception"}

        from shortcut_detect.benchmark.method_utils import (
            bias_direction_pca_dim_scores,
            nan_dim_scores,
            sis_dim_scores,
        )

        dim_score_fns: dict[str, Any] = {
            "probe": lambda: _probe_dim_scores(X, y, seed),
            "hbac": lambda: _hbac_dim_scores(method_results.get("hbac", {}), embedding_dim),
            "statistical": lambda: _statistical_dim_scores(
                method_results.get("statistical", {}), embedding_dim
            ),
            "geometric": lambda: _geometric_dim_scores(X, y),
            "bias_direction_pca": lambda: bias_direction_pca_dim_scores(
                method_results.get("bias_direction_pca", {}), embedding_dim
            ),
            "sis": lambda: sis_dim_scores(method_results.get("sis", {}), embedding_dim),
        }

        flags: list[int] = []
        rows: list[dict[str, Any]] = []
        for method in self.methods:
            flagged = _method_flag(method, method_results[method])
            flags.append(int(flagged))

            scores = dim_score_fns.get(method, lambda: nan_dim_scores(embedding_dim))()
            pred_dims = _top_k_from_scores(scores, self.shortcut_dims)
            precision, recall, f1 = _precision_recall_f1(pred_dims, true_dims)

            rows.append(
                {
                    "method": method,
                    "detected": int(flagged),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        convergence_count = int(sum(flags))
        for row in rows:
            row["convergence_count"] = convergence_count

        return rows

    # ------------------------------------------------------------------
    # Core sweep engine
    # ------------------------------------------------------------------
    def _run_sweep(
        self,
        sweep_param: str,
        param_values: list,
        build_kwargs: dict[str, Any],
        n_seeds: int,
    ) -> SweepResult:
        """Generic sweep engine shared by all public sweep methods."""
        seeds = self._seed_values(n_seeds)
        all_rows: list[dict[str, Any]] = []

        for pval in param_values:
            kwargs = dict(build_kwargs)
            kwargs[sweep_param] = pval

            for seed in seeds:
                X, y, true_dims = self._generate_synthetic_dataset(
                    n_samples=kwargs["n_samples"],
                    embedding_dim=kwargs["embedding_dim"],
                    shortcut_dims=self.shortcut_dims,
                    effect_size=kwargs["effect_size"],
                    group_ratio=kwargs["group_ratio"],
                    seed=seed,
                )
                run_rows = self._evaluate_single(X, y, true_dims, kwargs["embedding_dim"], seed)
                for row in run_rows:
                    row["param_value"] = pval
                    row["seed"] = seed
                all_rows.extend(run_rows)

        df = pd.DataFrame(all_rows)
        # Ensure canonical column order
        col_order = [
            "param_value",
            "method",
            "seed",
            "detected",
            "precision",
            "recall",
            "f1",
            "convergence_count",
        ]
        df = df[[c for c in col_order if c in df.columns]]

        return SweepResult(
            sweep_param=sweep_param,
            param_values=list(param_values),
            results_df=df,
        )

    # ------------------------------------------------------------------
    # Public sweep methods
    # ------------------------------------------------------------------

    def sweep_sample_size(
        self,
        sample_sizes: list[int],
        effect_size: float = 0.8,
        embedding_dim: int = 128,
        group_ratio: float = 0.5,
        n_seeds: int = 10,
    ) -> SweepResult:
        """B04 -- Sweep across sample sizes."""
        return self._run_sweep(
            sweep_param="n_samples",
            param_values=sample_sizes,
            build_kwargs={
                "n_samples": 0,  # placeholder, overwritten per value
                "embedding_dim": embedding_dim,
                "effect_size": effect_size,
                "group_ratio": group_ratio,
            },
            n_seeds=n_seeds,
        )

    def sweep_imbalance(
        self,
        group_ratios: list[float],
        effect_size: float = 0.8,
        n_samples: int = 1000,
        embedding_dim: int = 128,
        n_seeds: int = 10,
    ) -> SweepResult:
        """B05 -- Sweep across class imbalance ratios."""
        return self._run_sweep(
            sweep_param="group_ratio",
            param_values=group_ratios,
            build_kwargs={
                "n_samples": n_samples,
                "embedding_dim": embedding_dim,
                "effect_size": effect_size,
                "group_ratio": 0.5,  # placeholder
            },
            n_seeds=n_seeds,
        )

    def sweep_dimensionality(
        self,
        embedding_dims: list[int],
        effect_size: float = 0.8,
        n_samples: int = 1000,
        group_ratio: float = 0.5,
        n_seeds: int = 10,
    ) -> SweepResult:
        """B06 -- Sweep across embedding dimensionalities."""
        return self._run_sweep(
            sweep_param="embedding_dim",
            param_values=embedding_dims,
            build_kwargs={
                "n_samples": n_samples,
                "embedding_dim": 0,  # placeholder
                "effect_size": effect_size,
                "group_ratio": group_ratio,
            },
            n_seeds=n_seeds,
        )

    def sweep_custom(
        self,
        param_name: str,
        param_values: list,
        fixed_params: dict[str, Any],
        n_seeds: int = 10,
    ) -> SweepResult:
        """Generic sweep over an arbitrary generation parameter.

        ``fixed_params`` must supply the keys *n_samples*, *embedding_dim*,
        *effect_size*, and *group_ratio* (the swept key's value is
        overridden per iteration).
        """
        required = {"n_samples", "embedding_dim", "effect_size", "group_ratio"}
        missing = required - set(fixed_params)
        if missing:
            raise ValueError(f"fixed_params missing required keys: {missing}")
        return self._run_sweep(
            sweep_param=param_name,
            param_values=param_values,
            build_kwargs=dict(fixed_params),
            n_seeds=n_seeds,
        )
