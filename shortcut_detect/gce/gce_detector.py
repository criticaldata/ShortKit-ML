"""
GCE (Generalized Cross Entropy) bias detector.

Trains a biased classifier with GCE loss (q ≈ 0.7), records per-sample losses,
and labels high-loss samples as minority/bias-conflicting. High GCE loss indicates
samples the model finds hard to fit, often corresponding to minority or
bias-conflicting examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..detector_base import RiskLevel


@dataclass
class GCEDetectorReport:
    """Container for GCE detector metrics."""

    n_samples: int
    n_minority: int
    minority_ratio: float
    loss_mean: float
    loss_std: float
    loss_min: float
    loss_max: float
    threshold: float
    q: float
    risk_level: str
    notes: str
    reference: str


def _gce_loss_per_sample(probs: np.ndarray, y_true: np.ndarray, q: float) -> np.ndarray:
    """
    Generalized Cross Entropy loss per sample: L_i = (1 - p_i^q) / q,
    where p_i is the predicted probability for the true class.

    Args:
        probs: (n_samples, n_classes) predicted class probabilities
        y_true: (n_samples,) integer class labels in [0, n_classes-1]
        q: GCE parameter in (0, 1]; q≈0.7 downweights easy samples

    Returns:
        (n_samples,) per-sample GCE loss
    """
    if q <= 0 or q > 1:
        raise ValueError("q must be in (0, 1]")
    n = probs.shape[0]
    p_true = probs[np.arange(n), np.asarray(y_true, dtype=int)]
    p_true = np.clip(p_true, 1e-15, 1.0)
    return (1.0 - np.power(p_true, q)) / q


def _softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over last axis."""
    x = np.asarray(logits, dtype=float)
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _train_linear_gce(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    q: float,
    max_iter: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a linear classifier with GCE loss using L-BFGS-B.

    Args:
        X: (n_samples, n_features) features
        y: (n_samples,) integer labels in [0, n_classes-1]
        n_classes: number of classes
        q: GCE parameter
        max_iter: maximum optimization iterations
        random_state: random seed for init

    Returns:
        W: (n_features, n_classes) weights
        b: (n_classes,) bias
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    # Parameters: W (n_features, n_classes), b (n_classes)
    W = rng.standard_normal((n_features, n_classes)) * 0.01
    b = np.zeros(n_classes)
    theta0 = np.concatenate([W.ravel(), b])

    y_int = np.asarray(y, dtype=int)

    def objective(theta: np.ndarray) -> float:
        W_ = theta[: n_features * n_classes].reshape(n_features, n_classes)
        b_ = theta[n_features * n_classes :]
        logits = X @ W_ + b_
        probs = _softmax_stable(logits)
        losses = _gce_loss_per_sample(probs, y_int, q)
        return float(np.mean(losses))

    def gradient(theta: np.ndarray) -> np.ndarray:
        W_ = theta[: n_features * n_classes].reshape(n_features, n_classes)
        b_ = theta[n_features * n_classes :]
        logits = X @ W_ + b_
        probs = _softmax_stable(logits)
        n = probs.shape[0]
        # Gradient of GCE w.r.t. probs: d/dp_true (1 - p^q)/q = -p^(q-1)
        p_true = probs[np.arange(n), y_int]
        p_true = np.clip(p_true, 1e-15, 1.0)
        d_loss_d_p = np.zeros_like(probs)
        d_loss_d_p[np.arange(n), y_int] = -np.power(p_true, q - 1.0) / n
        # Backprop through softmax: d_loss/d_logits = probs * (d_loss_d_p - sum(d_loss_d_p * probs))
        sum_term = np.sum(d_loss_d_p * probs, axis=1, keepdims=True)
        d_logits = probs * (d_loss_d_p - sum_term)
        d_b = np.sum(d_logits, axis=0)
        d_W = X.T @ d_logits
        return np.concatenate([d_W.ravel(), d_b])

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=gradient,
        options={"maxiter": max_iter, "disp": False},
    )
    theta = res.x
    W = theta[: n_features * n_classes].reshape(n_features, n_classes)
    b = theta[n_features * n_classes :]
    return W, b


class GCEDetector:
    """
    Detect minority/bias-conflicting samples via Generalized Cross Entropy.

    Trains a linear classifier on embeddings with GCE loss (q ≈ 0.7). Samples
    with high per-sample GCE loss are flagged as minority or bias-conflicting,
    as they are harder for the biased classifier to fit.
    """

    def __init__(
        self,
        q: float = 0.7,
        loss_percentile_threshold: float = 90.0,
        max_iter: int = 500,
        random_state: int | None = 42,
    ) -> None:
        """
        Args:
            q: GCE parameter in (0, 1]. q≈0.7 downweights easy samples and
               emphasizes hard/minority ones. Smaller q is more robust but
               harder to optimize.
            loss_percentile_threshold: Samples with loss >= this percentile
               (0–100) are labeled as minority/bias-conflicting. Default 90.
            max_iter: Maximum iterations for training the linear classifier.
            random_state: Random seed for reproducibility.
        """
        if not 0 < q <= 1:
            raise ValueError("q must be in (0, 1]")
        if not 0 <= loss_percentile_threshold <= 100:
            raise ValueError("loss_percentile_threshold must be in [0, 100]")
        self.q = q
        self.loss_percentile_threshold = loss_percentile_threshold
        self.max_iter = max_iter
        self.random_state = random_state

        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.per_sample_losses_: np.ndarray | None = None
        self.is_minority_: np.ndarray | None = None
        self.loss_threshold_: float | None = None
        self.report_: GCEDetectorReport | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> GCEDetector:
        """
        Fit a GCE classifier and flag high-loss (minority/bias-conflicting) samples.

        Args:
            embeddings: (n_samples, n_features) embedding matrix
            labels: (n_samples,) integer or binary labels

        Returns:
            self
        """
        X = np.asarray(embeddings, dtype=float)
        y = np.asarray(labels)
        if X.ndim != 2:
            raise ValueError("embeddings must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("labels must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("embeddings and labels must have same length")

        # Map labels to 0, 1, ..., n_classes-1
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError("At least 2 distinct labels are required")
        y_int = np.searchsorted(classes, y)

        # Train linear classifier with GCE
        W, b = _train_linear_gce(X, y_int, n_classes, self.q, self.max_iter, self.random_state)
        self.coef_ = W
        self.intercept_ = b

        # Per-sample losses on training set
        logits = X @ W + b
        probs = _softmax_stable(logits)
        per_sample_losses = _gce_loss_per_sample(probs, y_int, self.q)
        self.per_sample_losses_ = per_sample_losses

        # Threshold: percentile of loss distribution
        threshold = float(np.percentile(per_sample_losses, self.loss_percentile_threshold))
        self.loss_threshold_ = threshold
        self.is_minority_ = per_sample_losses >= threshold

        n_samples = X.shape[0]
        n_minority = int(np.sum(self.is_minority_))
        minority_ratio = n_minority / n_samples if n_samples else 0.0
        loss_mean = float(np.mean(per_sample_losses))
        loss_std = float(np.std(per_sample_losses))
        loss_min = float(np.min(per_sample_losses))
        loss_max = float(np.max(per_sample_losses))

        risk_level, notes = _assess_risk(
            minority_ratio=minority_ratio,
            loss_mean=loss_mean,
            n_minority=n_minority,
        )

        self.report_ = GCEDetectorReport(
            n_samples=n_samples,
            n_minority=n_minority,
            minority_ratio=minority_ratio,
            loss_mean=loss_mean,
            loss_std=loss_std,
            loss_min=loss_min,
            loss_max=loss_max,
            threshold=threshold,
            q=self.q,
            risk_level=risk_level.value,
            notes=notes,
            reference="GCE bias detector (high-loss = minority/bias-conflicting)",
        )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class labels from embeddings."""
        self._ensure_fitted()
        logits = embeddings @ self.coef_ + self.intercept_
        pred_idx = np.argmax(logits, axis=1)
        return self.classes_[pred_idx]

    def get_minority_indices(self) -> np.ndarray:
        """Return indices of samples flagged as minority/bias-conflicting."""
        self._ensure_fitted()
        return np.where(self.is_minority_)[0]

    def _ensure_fitted(self) -> None:
        if self.coef_ is None or self.per_sample_losses_ is None:
            raise ValueError("GCEDetector must be fitted before using this method.")


def _assess_risk(
    minority_ratio: float,
    loss_mean: float,
    n_minority: int,
) -> tuple[RiskLevel, str]:
    """Compute risk level and notes from GCE metrics."""
    if minority_ratio > 0.25 or n_minority > 100:
        risk_level = RiskLevel.HIGH
        notes = (
            f"Many high-loss samples ({n_minority}) flagged as minority/bias-conflicting; "
            "strong shortcut or spurious correlation likely."
        )
    elif minority_ratio > 0.10 or n_minority > 20:
        risk_level = RiskLevel.MODERATE
        notes = (
            f"Moderate number of high-loss samples ({n_minority}); "
            "possible shortcut or subgroup imbalance."
        )
    else:
        risk_level = RiskLevel.LOW
        notes = (
            f"Few high-loss samples ({n_minority}); "
            "limited evidence of minority/bias-conflicting patterns."
        )
    return risk_level, notes
