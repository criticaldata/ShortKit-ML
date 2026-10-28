"""Synthetic data for the conditional shortcut-potential benchmark.

Y ~ Bernoulli(0.5) and A agrees with Y with probability ``rho``, so A and Y are
separate variables. The task signal sits in its own dimensions (shift of
+/- task_effect/2 by Y). The attribute signal sits in separate dimensions:

linear
    each group shifted by +/- delta/2 with unit noise, so the standardised mean
    difference between A groups is exactly delta (Cohen's d).
xor
    pairs of dimensions carry the attribute in the sign of their product. Each
    pair gets component signs (u1, u2) with u1 random and u2 = u1 for A = 1,
    u2 = -u1 for A = 0, and is shifted by (u1, u2) * delta/2. Both groups
    have zero mean in every dimension; given the partner component, the
    groups differ by exactly delta standard deviations.
variance
    equal means; the attribute dimensions have standard deviation 1 + delta for
    A = 1 and 1 for A = 0.

The ground truth is I(E; A | Y) > 0 exactly when delta > 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VARIANTS = ("linear", "xor", "variance")


@dataclass(frozen=True)
class ConditionalSyntheticDataset:
    embeddings: np.ndarray
    task_labels: np.ndarray
    attribute: np.ndarray
    attribute_dims: np.ndarray
    task_dims: np.ndarray
    delta: float
    rho: float
    variant: str
    seed: int

    @property
    def has_shortcut_potential(self) -> bool:
        return self.delta > 0


def generate_conditional_synthetic(
    *,
    n_samples: int = 1000,
    embedding_dim: int = 128,
    delta: float = 0.5,
    rho: float = 0.7,
    variant: str = "linear",
    attribute_dims: int = 4,
    task_dims: int = 4,
    task_effect: float = 1.0,
    seed: int = 0,
) -> ConditionalSyntheticDataset:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if attribute_dims + task_dims > embedding_dim:
        raise ValueError("attribute_dims + task_dims cannot exceed embedding_dim")
    if variant == "xor" and attribute_dims % 2:
        raise ValueError("xor needs an even number of attribute dimensions")

    rng = np.random.default_rng(seed)
    n = n_samples
    y = (rng.random(n) < 0.5).astype(np.int64)
    a = np.where(rng.random(n) < rho, y, 1 - y).astype(np.int64)
    X = rng.standard_normal((n, embedding_dim))

    t_dims = np.arange(task_dims)
    a_dims = np.arange(task_dims, task_dims + attribute_dims)
    X[:, t_dims] += (2 * y[:, None] - 1) * task_effect / 2.0

    if delta > 0:
        sa = (2 * a - 1).astype(float)
        if variant == "linear":
            X[:, a_dims] += sa[:, None] * delta / 2.0
        elif variant == "xor":
            for j in range(0, attribute_dims, 2):
                u1 = rng.choice([-1.0, 1.0], size=n)
                u2 = u1 * sa
                X[:, a_dims[j]] += u1 * delta / 2.0
                X[:, a_dims[j + 1]] += u2 * delta / 2.0
        else:
            X[np.ix_(a == 1, a_dims)] *= 1.0 + delta

    return ConditionalSyntheticDataset(
        embeddings=X.astype(np.float32),
        task_labels=y,
        attribute=a,
        attribute_dims=a_dims,
        task_dims=t_dims,
        delta=float(delta),
        rho=float(rho),
        variant=variant,
        seed=seed,
    )
