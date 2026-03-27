"""
Common metrics used to evaluate probes.
Expose a registry so probe.score can accept metric names.
"""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mutual_info_score,
    roc_auc_score,
)


def _safe_roc_auc(y_true, y_pred_proba):
    try:
        return roc_auc_score(y_true, y_pred_proba)
    except Exception:
        return float("nan")


def mutual_information(y_true, y_pred):
    return mutual_info_score(y_true.astype(int), np.round(y_pred).astype(int))


metrics_registry: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
    "f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="weighted")),
    "roc_auc": lambda y_true, y_pred: float(_safe_roc_auc(y_true, y_pred)),
    "mse": lambda y_true, y_pred: float(mean_squared_error(y_true, y_pred)),
    "mutual_info": lambda y_true, y_pred: float(mutual_information(y_true, y_pred)),
}
