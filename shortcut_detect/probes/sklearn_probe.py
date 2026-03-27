# shortcut_detect/ml_probe.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from shortcut_detect.detector_base import DetectorBase
from shortcut_detect.utils import validate_embeddings_labels

try:
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, label_binarize
except Exception as e:  # pragma: no cover
    raise ImportError(
        "SKLearnProbe requires scikit-learn. Install it with: pip install scikit-learn"
    ) from e


MetricName = Literal["accuracy", "f1", "precision", "recall", "roc_auc"]
EvaluationName = Literal["train", "holdout", "cv"]


def _is_binary(y: np.ndarray) -> bool:
    return np.unique(y).shape[0] == 2


def _score_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    metric: MetricName,
    average: str,
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, average=average))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    if metric == "roc_auc":
        # Needs probabilities or decision scores.
        if y_score is None:
            raise ValueError("roc_auc requires predict_proba or decision_function.")
        # Binary: y_score is (n,) for positive class
        # Multiclass: y_score should be (n, K)
        if y_score.ndim == 1:
            return float(roc_auc_score(y_true, y_score))
        # multiclass one-vs-rest
        classes = np.unique(y_true)
        Y = label_binarize(y_true, classes=classes)
        return float(roc_auc_score(Y, y_score, average="macro", multi_class="ovr"))
    raise ValueError(f"Unknown metric: {metric}")


def _predict_scores(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = np.asarray(proba)
        # For binary, many metrics expect positive-class probability as (n,)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return np.asarray(scores)
    return None


@dataclass(frozen=True)
class MLProbeConfig:
    metric: MetricName = "f1"
    threshold: float = 0.70
    average: str = "macro"  # for f1/precision/recall in multiclass
    evaluation: EvaluationName = "holdout"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 0


class SKLearnProbe(DetectorBase):
    """Shortcut detector based on training a classifier to predict a demographic target.

    Idea:
      - Train a probe classifier to predict a sensitive/demographic attribute y from embeddings X.
      - If the probe performs above a user-defined threshold on a metric (e.g., F1),
        treat this as evidence that embeddings encode the attribute (potential shortcut).

    Parameters
    ----------
    estimator:
        Any scikit-learn estimator supporting fit/predict (optionally predict_proba or decision_function).
        If None, uses a standardized LogisticRegression.
    metric:
        One of: "accuracy", "f1", "precision", "recall", "roc_auc".
    threshold:
        Shortcut is detected if metric_value > threshold.
    average:
        Averaging strategy for multiclass f1/precision/recall ("macro", "micro", "weighted").
        For binary problems, "binary" is used automatically for these metrics.
    evaluation:
        "holdout" (train/test split), "cv" (StratifiedKFold cross-validation), or "train" (no splitting).
    test_size:
        Used for holdout split.
    cv_folds:
        Used for CV.
    random_state:
        Reproducibility for splitting.

    Fit inputs
    ----------
    embeddings: np.ndarray, shape (n_samples, n_features)
    target: np.ndarray, shape (n_samples,)
        Demographic/sensitive attribute labels (e.g., gender).
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        metric: MetricName = "f1",
        threshold: float = 0.70,
        average: str = "macro",
        evaluation: EvaluationName = "holdout",
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 0,
    ):
        super().__init__(method="ml_probe")

        if estimator is None:
            # Reasonable default probe: scale then logistic regression
            estimator = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                            random_state=random_state,
                        ),
                    ),
                ]
            )

        self.estimator = estimator
        self.config = MLProbeConfig(
            metric=metric,
            threshold=float(threshold),
            average=average,
            evaluation=evaluation,
            test_size=float(test_size),
            cv_folds=int(cv_folds),
            random_state=int(random_state),
        )

        # Fitted artifacts
        self.estimator_: BaseEstimator | None = None
        self.metric_value_: float | None = None
        self.y_true_eval_: np.ndarray | None = None
        self.y_pred_eval_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, target: np.ndarray) -> SKLearnProbe:
        X, y = validate_embeddings_labels(embeddings, target, min_samples=4)

        # Encode string labels to integers so sklearn metrics work correctly
        if y.dtype.kind in ("U", "S", "O"):  # Unicode, byte-string, or object
            from sklearn.preprocessing import LabelEncoder

            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)

        classes, counts = np.unique(y, return_counts=True)
        n_classes = int(classes.shape[0])

        # Average handling
        metric = self.config.metric
        average = self.config.average
        if metric in {"f1", "precision", "recall"} and _is_binary(y):
            average_used = "binary"
        else:
            average_used = average

        # Run evaluation
        if self.config.evaluation == "holdout":
            metric_value, report = self._eval_holdout(X, y, metric, average_used)
        elif self.config.evaluation == "cv":
            metric_value, report = self._eval_cv(X, y, metric, average_used)
        elif self.config.evaluation == "train":
            metric_value, report = self._eval_train(X, y, metric, average_used)
        else:
            raise ValueError(f"Unknown evaluation={self.config.evaluation}")

        self.metric_value_ = float(metric_value)
        shortcut = bool(self.metric_value_ > self.config.threshold)
        self.shortcut_detected_ = shortcut

        risk_level: str | Literal["low", "moderate", "high", "unknown"]
        # Simple mapping; users can override via threshold tuning
        if metric_value >= max(self.config.threshold, 0.85):
            risk_level = "high"
        elif metric_value >= self.config.threshold:
            risk_level = "moderate"
        else:
            risk_level = "low"

        notes = (
            "Trained a probe classifier to predict the provided target from embeddings. "
            f"shortcut_detected is True when {self.config.metric} > threshold."
        )

        self._set_results(
            shortcut_detected=shortcut,
            risk_level=risk_level,
            metrics={
                "metric": self.config.metric,
                "metric_value": self.metric_value_,
                "threshold": self.config.threshold,
                "evaluation": self.config.evaluation,
                "n_classes": n_classes,
            },
            notes=notes,
            metadata={
                "class_distribution": {
                    str(c): int(n) for c, n in zip(classes, counts, strict=False)
                },
                "average": average_used,
                "test_size": self.config.test_size if self.config.evaluation == "holdout" else None,
                "cv_folds": self.config.cv_folds if self.config.evaluation == "cv" else None,
                "estimator": self._estimator_name(),
                "random_state": self.config.random_state,
            },
            report=report,
        )

        self._is_fitted = True
        return self

    def _estimator_name(self) -> str:
        est = self.estimator
        if isinstance(est, Pipeline):
            return "Pipeline(" + "->".join([name for name, _ in est.steps]) + ")"
        return est.__class__.__name__

    def _eval_holdout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: MetricName,
        average_used: str,
    ) -> tuple[float, dict[str, Any]]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state,
        )
        est = self._clone_estimator()
        est.fit(X_tr, y_tr)

        y_pred = np.asarray(est.predict(X_te))
        y_score = _predict_scores(est, X_te) if metric == "roc_auc" else _predict_scores(est, X_te)
        self.y_true_eval_ = np.asarray(y_te)
        self.y_pred_eval_ = np.asarray(y_pred)

        metric_value = _score_metric(y_te, y_pred, y_score, metric, average_used)

        # Persist the fitted estimator from holdout (useful for later inference/debug)
        self.estimator_ = est

        report: dict[str, Any] = {
            "protocol": "holdout",
            "test_size": self.config.test_size,
            "metric": metric,
            "average": average_used,
            "metric_value": float(metric_value),
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
        }
        return float(metric_value), report

    def _eval_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: MetricName,
        average_used: str,
    ) -> tuple[float, dict[str, Any]]:
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        fold_scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            est = self._clone_estimator()
            est.fit(X_tr, y_tr)

            y_pred = np.asarray(est.predict(X_te))
            y_score = (
                _predict_scores(est, X_te) if metric == "roc_auc" else _predict_scores(est, X_te)
            )

            s = _score_metric(y_te, y_pred, y_score, metric, average_used)
            fold_scores.append(float(s))

        fold_scores_arr = np.asarray(fold_scores, dtype=float)
        metric_value = float(fold_scores_arr.mean())

        # For CV, we also fit a final estimator on all data for optional later use
        final_est = self._clone_estimator()
        final_est.fit(X, y)
        self.estimator_ = final_est

        report: dict[str, Any] = {
            "protocol": "cv",
            "cv_folds": int(self.config.cv_folds),
            "metric": metric,
            "average": average_used,
            "fold_scores": fold_scores,
            "mean_score": metric_value,
            "std_score": float(fold_scores_arr.std(ddof=1)) if len(fold_scores) > 1 else 0.0,
            "n_samples": int(X.shape[0]),
        }
        return metric_value, report

    def _eval_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: MetricName,
        average_used: str,
    ) -> tuple[float, dict[str, Any]]:
        est = self._clone_estimator()
        est.fit(X, y)

        y_pred = np.asarray(est.predict(X))
        y_score = _predict_scores(est, X) if metric == "roc_auc" else _predict_scores(est, X)
        self.y_true_eval_ = np.asarray(y)
        self.y_pred_eval_ = np.asarray(y_pred)

        metric_value = _score_metric(y, y_pred, y_score, metric, average_used)
        self.estimator_ = est

        report: dict[str, Any] = {
            "protocol": "train",
            "metric": metric,
            "average": average_used,
            "metric_value": float(metric_value),
            "n_samples": int(X.shape[0]),
        }
        return float(metric_value), report

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for embeddings (requires prior fit)."""
        self._ensure_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features). Got shape={X.shape}.")
        return np.asarray(self.estimator_.predict(X))

    def _clone_estimator(self) -> BaseEstimator:
        # Prefer sklearn.clone, but avoid importing clone if not needed.
        try:
            from sklearn.base import clone  # type: ignore

            return clone(self.estimator)
        except Exception:
            # Fallback: best-effort shallow copy; may fail for some estimators.
            import copy

            return copy.deepcopy(self.estimator)
