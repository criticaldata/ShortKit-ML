"""
Utility helpers: reproducibility, train/test split, batching helpers.
"""

import os
import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across numpy, random, and (if used) torch."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_embeddings_labels(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    min_samples: int = 4,
    min_classes: int = 0,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 2D embeddings + 1D labels shape contract, minimum sample count,
    finite values, and minimum class count.

    Parameters
    ----------
    embeddings : array-like
        2D array of shape (n_samples, n_features).
    labels : array-like
        1D array of shape (n_samples,).
    min_samples : int
        Minimum number of samples required.
    min_classes : int
        Minimum number of unique classes in *labels*. Set to 2 for methods
        that require at least two classes. 0 disables the check.
    check_finite : bool
        If True, raise on NaN or Inf values in embeddings.
    """
    X = np.asarray(embeddings, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={X.shape}")
    if X.shape[1] == 0:
        raise ValueError("embeddings must have at least one feature (column).")
    if y.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"embeddings and labels must have the same number of samples: "
            f"{X.shape[0]} != {y.shape[0]}"
        )
    if X.shape[0] < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {X.shape[0]}.")
    if check_finite and not np.all(np.isfinite(X)):
        raise ValueError(
            "embeddings contain NaN or infinite values. "
            "Clean your data before running detection."
        )
    if min_classes > 0:
        n_unique = len(np.unique(y))
        if n_unique < min_classes:
            raise ValueError(
                f"At least {min_classes} distinct classes required in labels, " f"got {n_unique}."
            )
    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple train/test split."""
    n = X.shape[0]
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float in (0,1)")
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def batch_generator(
    X: np.ndarray, y: np.ndarray, batch_size: int = 128, shuffle: bool = True, seed: int = 0
):
    """Yield (X_batch, y_batch) for training."""
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]
