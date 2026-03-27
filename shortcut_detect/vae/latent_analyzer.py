"""
Latent space analysis for VAE shortcut detection.

Computes MPWD (max pairwise Wasserstein distance) and predictiveness per dimension.
Adapted from Müller et al., Fraunhofer-AISEC/shortcut-detection-vae.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance


def compute_mpwd_per_dimension(
    latents: np.ndarray,
    labels: np.ndarray,
    latent_dim: int,
    num_classes: int,
) -> np.ndarray:
    """
    Max pairwise Wasserstein distance (MPWD) per latent dimension across classes.

    For each dimension, compute Wasserstein distance between all class pairs;
    return the max per dimension. High MPWD indicates the dimension separates classes.

    Args:
        latents: (n_samples, latent_dim) encoded latent codes
        labels: (n_samples,) class labels in [0, num_classes-1]
        latent_dim: number of latent dimensions
        num_classes: number of classes

    Returns:
        (latent_dim,) array of max pairwise Wasserstein distances per dimension
    """
    distances = np.zeros((latent_dim, num_classes, num_classes))
    for dim in range(latent_dim):
        for cls_1 in range(num_classes):
            for cls_2 in range(num_classes):
                val1 = latents[labels == cls_1, dim]
                val2 = latents[labels == cls_2, dim]
                if len(val1) > 0 and len(val2) > 0:
                    distances[dim, cls_1, cls_2] = wasserstein_distance(val1, val2)
    return distances.max(axis=(1, 2))


def compute_predictiveness_per_dimension(
    classifier: torch.nn.Module,
    latent_dim: int,
) -> np.ndarray:
    """
    Per-dimension predictiveness from classifier linear layer weights.

    Predictiveness = sum of absolute weights for each latent dimension across classes.
    High predictiveness indicates the dimension is used for classification (candidate shortcut).

    Args:
        classifier: VAEClassifier with .fc (Linear) layer
        latent_dim: number of latent dimensions

    Returns:
        (latent_dim,) array of predictiveness scores
    """
    fc = classifier.fc
    if not hasattr(fc, "weight"):
        raise ValueError("Classifier must have .fc.weight (Linear layer)")
    weights = fc.weight.detach().cpu().numpy()  # (num_classes, latent_dim)
    predictiveness = np.abs(weights).sum(axis=0)
    return np.asarray(predictiveness, dtype=np.float64)


def rank_candidate_dimensions(
    predictiveness: np.ndarray,
    mpwd: np.ndarray,
    predictiveness_threshold: float,
) -> tuple[np.ndarray, list[int]]:
    """
    Rank latent dimensions as shortcut candidates.

    A dimension is flagged if its predictiveness exceeds the threshold.
    Returns sorted indices (descending by predictiveness) and flagged indices.

    Args:
        predictiveness: (latent_dim,) predictiveness scores
        mpwd: (latent_dim,) MPWD scores
        predictiveness_threshold: threshold for flagging

    Returns:
        sorted_indices: indices sorted by predictiveness (high first)
        flagged_indices: indices where predictiveness >= threshold
    """
    sorted_indices = np.argsort(predictiveness)[::-1]
    flagged_indices = [
        i for i in range(len(predictiveness)) if predictiveness[i] >= predictiveness_threshold
    ]
    return np.asarray(sorted_indices), flagged_indices
