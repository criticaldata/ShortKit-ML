"""
ProbePipeline: orchestration utilities for training and evaluation.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from ..metrics import metrics_registry
from ..utils import train_test_split


def evaluate_probe_cv(
    probe_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 0,
    metric: str = "accuracy",
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Cross-validate a probe."""
    if metric not in metrics_registry:
        raise ValueError(f"Unknown metric {metric}")
    n = X.shape[0]
    if n_splits < 2:
        raise ValueError("n_splits >= 2 required")

    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    if shuffle:
        rng.shuffle(idx)
    fold_sizes = [(n // n_splits) + (1 if i < (n % n_splits) else 0) for i in range(n_splits)]
    starts = []
    s = 0
    for fs in fold_sizes:
        starts.append((s, s + fs))
        s += fs

    def run_fold(fold):
        s, e = starts[fold]
        test_idx = idx[s:e]
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        probe = probe_factory()
        probe.fit(X_train, y_train)
        score = probe.score(X_test, y_test, metric=metric)
        return float(score)

    results = Parallel(n_jobs=n_jobs)(delayed(run_fold)(i) for i in range(n_splits))
    results = np.array(results)
    return {
        "metric": metric,
        "mean": float(results.mean()),
        "std": float(results.std()),
        "folds": results.tolist(),
    }


def train_test_pipeline(
    probe: Any,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
    metric: str = "accuracy",
    shuffle: bool = True,
) -> dict[str, Any]:
    """Fit the given probe on a train split and evaluate on a test split."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, shuffle=shuffle, seed=seed)
    probe.fit(Xtr, ytr)
    train_score = probe.score(Xtr, ytr, metric=metric)
    test_score = probe.score(Xte, yte, metric=metric)
    return {"train_score": float(train_score), "test_score": float(test_score), "probe": probe}
