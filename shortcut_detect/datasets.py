"""
Synthetic dataset generators for testing and examples.

These functions generate embeddings with known shortcuts for validation
and demonstration purposes.
"""

import numpy as np


def generate_linear_shortcut(
    n_samples: int = 1000, embedding_dim: int = 128, shortcut_dims: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate embeddings with linear shortcuts in first N dimensions.

    Creates binary classification data where the first `shortcut_dims`
    dimensions have strong linear separation between classes.

    Args:
        n_samples: Number of samples to generate
        embedding_dim: Total embedding dimensionality
        shortcut_dims: Number of dimensions with shortcuts (first N dims)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (embeddings, labels):
            - embeddings: (n_samples, embedding_dim) array
            - labels: (n_samples,) binary labels (0 or 1)

    Example:
        >>> embeddings, labels = generate_linear_shortcut(
        ...     n_samples=1000,
        ...     embedding_dim=128,
        ...     shortcut_dims=5
        ... )
        >>> embeddings.shape
        (1000, 128)
    """
    rng = np.random.RandomState(seed)

    # Binary labels
    labels = rng.randint(0, 2, size=n_samples)

    # Initialize embeddings
    embeddings = rng.randn(n_samples, embedding_dim)

    # Add strong linear separation in first shortcut_dims dimensions
    for i in range(shortcut_dims):
        # Class 0: negative values, Class 1: positive values
        embeddings[labels == 0, i] = rng.randn(np.sum(labels == 0)) - 3
        embeddings[labels == 1, i] = rng.randn(np.sum(labels == 1)) + 3

    return embeddings, labels


def generate_nonlinear_shortcut(
    n_samples: int = 1000, embedding_dim: int = 128, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate embeddings with non-linear (radial) shortcuts.

    Creates binary classification data where classes form concentric
    circles in 2D space (first two dimensions).

    Args:
        n_samples: Number of samples to generate
        embedding_dim: Total embedding dimensionality
        seed: Random seed for reproducibility

    Returns:
        Tuple of (embeddings, labels):
            - embeddings: (n_samples, embedding_dim) array
            - labels: (n_samples,) binary labels (0 or 1)

    Example:
        >>> embeddings, labels = generate_nonlinear_shortcut(
        ...     n_samples=800,
        ...     embedding_dim=64
        ... )
        >>> embeddings.shape
        (800, 64)
    """
    rng = np.random.RandomState(seed)

    # Binary labels
    labels = rng.randint(0, 2, size=n_samples)

    # Initialize with random embeddings
    embeddings = rng.randn(n_samples, embedding_dim)

    # Create concentric circles in first 2 dimensions
    angles = rng.uniform(0, 2 * np.pi, n_samples)

    # Class 0: inner circle (radius ~1)
    # Class 1: outer circle (radius ~3)
    radius_class0 = 1.0 + rng.randn(np.sum(labels == 0)) * 0.2
    radius_class1 = 3.0 + rng.randn(np.sum(labels == 1)) * 0.2

    embeddings[labels == 0, 0] = radius_class0 * np.cos(angles[labels == 0])
    embeddings[labels == 0, 1] = radius_class0 * np.sin(angles[labels == 0])
    embeddings[labels == 1, 0] = radius_class1 * np.cos(angles[labels == 1])
    embeddings[labels == 1, 1] = radius_class1 * np.sin(angles[labels == 1])

    return embeddings, labels


def generate_multiclass_shortcut(
    n_samples: int = 1500, embedding_dim: int = 64, n_classes: int = 3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate multi-class embeddings with shortcuts.

    Creates multi-class classification data where each class has
    distinct mean in the first few dimensions.

    Args:
        n_samples: Number of samples to generate
        embedding_dim: Total embedding dimensionality
        n_classes: Number of classes
        seed: Random seed for reproducibility

    Returns:
        Tuple of (embeddings, labels):
            - embeddings: (n_samples, embedding_dim) array
            - labels: (n_samples,) labels in range [0, n_classes-1]

    Example:
        >>> embeddings, labels = generate_multiclass_shortcut(
        ...     n_samples=1500,
        ...     n_classes=3
        ... )
        >>> len(np.unique(labels))
        3
    """
    rng = np.random.RandomState(seed)

    # Multi-class labels
    labels = rng.randint(0, n_classes, size=n_samples)

    # Initialize embeddings
    embeddings = rng.randn(n_samples, embedding_dim)

    # Each class gets a distinct mean in first 3 dimensions
    class_means = rng.randn(n_classes, 3) * 5  # Well-separated means

    for c in range(n_classes):
        mask = labels == c
        for dim in range(3):
            embeddings[mask, dim] += class_means[c, dim]

    return embeddings, labels


def generate_no_shortcut(
    n_samples: int = 1000, embedding_dim: int = 64, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random embeddings with NO shortcuts (negative control).

    Creates completely random data with no correlation between
    embeddings and labels.

    Args:
        n_samples: Number of samples to generate
        embedding_dim: Total embedding dimensionality
        seed: Random seed for reproducibility

    Returns:
        Tuple of (embeddings, labels):
            - embeddings: (n_samples, embedding_dim) array - pure random
            - labels: (n_samples,) binary labels (0 or 1) - pure random

    Example:
        >>> embeddings, labels = generate_no_shortcut(n_samples=500)
        >>> # Should show no shortcuts when tested
    """
    rng = np.random.RandomState(seed)

    # Completely random
    embeddings = rng.randn(n_samples, embedding_dim)
    labels = rng.randint(0, 2, size=n_samples)

    return embeddings, labels


def generate_linear_shortcut_with_group_labels(
    n_samples: int = 800,
    embedding_dim: int = 20,
    signal_dim: int = 0,
    hard_group_noise: float = 2.5,
    easy_group_noise: float = 0.3,
    seed: int = 42,
):
    """
    Construct a dataset with two groups:
      - group 0: easier classification (low noise)
      - group 1: harder classification (high noise)

    Labels are generated from a linear separator in one dimension, then noise added.
    """
    rng = np.random.RandomState(seed)

    # balanced groups
    g = rng.randint(0, 2, size=n_samples).astype(np.int64)

    X = rng.randn(n_samples, embedding_dim).astype(np.float32)

    # base signal
    w = np.zeros(embedding_dim, dtype=np.float32)
    w[signal_dim] = 2.0  # strong linear signal in one dimension

    # group-dependent noise
    noise = rng.randn(n_samples).astype(np.float32)
    noise_scale = np.where(g == 1, hard_group_noise, easy_group_noise).astype(np.float32)

    logits = X @ w + noise_scale * noise
    y = (logits > 0).astype(np.int64)

    return X, y, g
