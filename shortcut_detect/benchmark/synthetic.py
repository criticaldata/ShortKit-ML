"""Synthetic benchmark data generators with reproducible shortcut effect sizes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SHORTCUT_EFFECT_SIZE = 0.2
MAX_SHORTCUT_EFFECT_SIZE = 2.0


@dataclass(frozen=True)
class SyntheticShortcutConfig:
    """Configuration for reproducible synthetic shortcut embeddings."""

    n_samples: int = 1000
    embedding_dim: int = 128
    shortcut_dims: int = 5
    effect_size: float = 1.0
    positive_class_probability: float = 0.5
    noise_std: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if self.shortcut_dims <= 0:
            raise ValueError("shortcut_dims must be > 0")
        if self.shortcut_dims > self.embedding_dim:
            raise ValueError("shortcut_dims cannot exceed embedding_dim")
        if not (MIN_SHORTCUT_EFFECT_SIZE <= self.effect_size <= MAX_SHORTCUT_EFFECT_SIZE):
            raise ValueError(
                "effect_size must be between "
                f"{MIN_SHORTCUT_EFFECT_SIZE} and {MAX_SHORTCUT_EFFECT_SIZE}"
            )
        if not (0.0 < self.positive_class_probability < 1.0):
            raise ValueError("positive_class_probability must be in (0, 1)")
        if self.noise_std <= 0.0:
            raise ValueError("noise_std must be > 0")


@dataclass(frozen=True)
class SyntheticShortcutDataset:
    """Synthetic embeddings and ground-truth shortcut annotations."""

    embeddings: np.ndarray
    labels: np.ndarray
    shortcut_dim_labels: np.ndarray
    shortcut_dim_indices: np.ndarray
    effect_size: float
    seed: int


def generate_parametric_shortcut_dataset(
    *,
    n_samples: int = 1000,
    embedding_dim: int = 128,
    shortcut_dims: int = 5,
    effect_size: float = 1.0,
    positive_class_probability: float = 0.5,
    noise_std: float = 1.0,
    seed: int = 42,
) -> SyntheticShortcutDataset:
    """
    Generate embeddings with a controlled linear shortcut strength.

    The first ``shortcut_dims`` embedding dimensions are shifted by ``effect_size``
    in opposite directions for the two classes. The remaining dimensions are pure noise.
    Ground-truth shortcut labels are returned as both a boolean mask and explicit indices.
    """
    config = SyntheticShortcutConfig(
        n_samples=n_samples,
        embedding_dim=embedding_dim,
        shortcut_dims=shortcut_dims,
        effect_size=effect_size,
        positive_class_probability=positive_class_probability,
        noise_std=noise_std,
        seed=seed,
    )
    config.validate()

    rng = np.random.RandomState(config.seed)
    labels = (rng.rand(config.n_samples) < config.positive_class_probability).astype(np.int64)
    embeddings = (
        rng.normal(
            loc=0.0,
            scale=config.noise_std,
            size=(config.n_samples, config.embedding_dim),
        )
    ).astype(np.float32)

    shortcut_dim_indices = np.arange(config.shortcut_dims, dtype=np.int64)
    shortcut_dim_labels = np.zeros(config.embedding_dim, dtype=bool)
    shortcut_dim_labels[shortcut_dim_indices] = True

    negative_mask = labels == 0
    positive_mask = ~negative_mask
    embeddings[np.ix_(negative_mask, shortcut_dim_indices)] -= config.effect_size
    embeddings[np.ix_(positive_mask, shortcut_dim_indices)] += config.effect_size

    return SyntheticShortcutDataset(
        embeddings=embeddings,
        labels=labels,
        shortcut_dim_labels=shortcut_dim_labels,
        shortcut_dim_indices=shortcut_dim_indices,
        effect_size=config.effect_size,
        seed=config.seed,
    )
