"""Parametric synthetic data generator for benchmark experiments.

Generates embeddings with controlled shortcut effect sizes for evaluating
detection methods. Effect sizes follow Cohen's d convention:
  0.0 = no shortcut (null/clean)
  0.2 = subtle
  0.5 = moderate
  0.8 = strong
  1.2-2.0 = very strong

Three generation modes are available:

* :meth:`SyntheticGenerator.generate` — standard independent shift: each
  shortcut dimension is shifted by ±effect_size independently.

* :meth:`SyntheticGenerator.generate_correlated` — harder correlated
  variant: the shortcut dimensions form a correlated block (compound
  symmetry), so the signal is spread across jointly-varying dimensions.
  Individual univariate tests lose power; multivariate or geometry-based
  methods are needed.

* :meth:`SyntheticGenerator.generate_distributed` — harder distributed
  variant: the effect is spread uniformly across all embedding dimensions
  at a per-dimension shift of ``effect_size * sqrt(shortcut_dims /
  embedding_dim)``, preserving total signal energy while making no single
  dimension stand out.

Usage:
    from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator

    gen = SyntheticGenerator(n_samples=1000, embedding_dim=128, shortcut_dims=5, seed=42)
    result = gen.generate(effect_size=0.8)
    result_hard = gen.generate_correlated(effect_size=0.8, correlation=0.7)
    result_dist = gen.generate_distributed(effect_size=0.8)
    # result.embeddings, result.labels, result.group_labels, result.shortcut_dims
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticResult:
    """Result container for parametric synthetic data generation.

    Attributes:
        embeddings: Array of shape ``(n_samples, embedding_dim)`` with generated
            feature vectors.
        labels: Array of shape ``(n_samples,)`` with binary class labels (0 or 1).
        group_labels: Array of shape ``(n_samples,)`` with binary group
            membership (0 or 1).  Group membership correlates with labels and
            determines the shortcut signal in the designated dimensions.
        shortcut_dims: List of dimension indices that carry the shortcut signal.
    """

    embeddings: np.ndarray
    labels: np.ndarray
    group_labels: np.ndarray
    shortcut_dims: list[int]


class SyntheticGenerator:
    """Parametric synthetic data generator with controlled effect sizes.

    The generator creates embeddings where a subset of dimensions (the
    *shortcut dimensions*) have their group means separated by
    ``effect_size`` standard deviations.  Non-shortcut dimensions are
    filled with standard-normal noise.  Labels correlate with group
    membership so that group 0 is predominantly label 0 and group 1 is
    predominantly label 1.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    embedding_dim : int
        Total number of embedding dimensions.
    shortcut_dims : int
        Number of leading dimensions that carry the shortcut signal.
    group_ratio : float
        Fraction of samples assigned to the *majority* group (group 0).
        Use 0.5 for balanced groups, higher values (e.g. 0.7, 0.9) for
        imbalanced groups.
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        embedding_dim: int = 128,
        shortcut_dims: int = 5,
        group_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if shortcut_dims < 0 or shortcut_dims > embedding_dim:
            raise ValueError(f"shortcut_dims must be in [0, embedding_dim], got {shortcut_dims}")
        if not (0.0 < group_ratio < 1.0):
            raise ValueError("group_ratio must be in (0, 1)")

        self.n_samples = n_samples
        self.embedding_dim = embedding_dim
        self.shortcut_dims = shortcut_dims
        self.group_ratio = group_ratio
        self.seed = seed

    def generate(self, effect_size: float = 0.8) -> SyntheticResult:
        """Generate a synthetic dataset with the given shortcut *effect_size*.

        The approach mirrors
        :pymethod:`PaperBenchmarkRunner._generate_synthetic_dataset` for
        consistency: group labels are drawn from a Bernoulli distribution
        controlled by ``group_ratio``, embeddings start as standard-normal
        noise, and shortcut dimensions have their means shifted by
        ``+/- effect_size`` for groups 1 and 0 respectively.

        Parameters
        ----------
        effect_size : float
            Magnitude of the group-mean shift applied to the shortcut
            dimensions.  ``0.0`` produces no shortcut; ``2.0`` produces a
            very strong shortcut.

        Returns
        -------
        SyntheticResult
            Named container with ``embeddings``, ``labels``,
            ``group_labels``, and ``shortcut_dims``.
        """
        if effect_size < 0.0:
            raise ValueError("effect_size must be >= 0")

        rng = np.random.RandomState(self.seed)

        # Group assignment: group 0 is the majority group.
        # group_ratio is the probability of being in group 0 (majority).
        group_labels = (rng.rand(self.n_samples) > self.group_ratio).astype(np.int64)

        # Labels correlate with group membership (group 0 -> label 0,
        # group 1 -> label 1).  This matches the paper_runner convention
        # where ``y`` directly drives the shortcut shift.
        labels = group_labels.copy()

        # Base embeddings: standard-normal noise.
        embeddings = rng.randn(self.n_samples, self.embedding_dim).astype(np.float32)

        # Shortcut dimension indices.
        true_dims = list(range(self.shortcut_dims))

        # Apply effect-size shift in shortcut dimensions.
        if effect_size > 0.0:
            for dim in true_dims:
                embeddings[labels == 0, dim] -= effect_size
                embeddings[labels == 1, dim] += effect_size

        return SyntheticResult(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            shortcut_dims=true_dims,
        )

    def generate_correlated(
        self, effect_size: float = 0.8, correlation: float = 0.7
    ) -> SyntheticResult:
        """Generate a harder synthetic dataset with correlated shortcut dimensions.

        Unlike :meth:`generate`, where each shortcut dimension shifts
        independently, here the ``shortcut_dims`` dimensions form a
        correlated block with uniform off-diagonal correlation ``rho``.
        The group means are still shifted by ``±effect_size``, but the
        inter-dimensional correlation reduces the effective degrees of
        freedom and makes univariate per-dimension tests less powerful.
        Methods that capture joint distributional structure (e.g. MMD,
        geometry-based probes) retain sensitivity.

        Parameters
        ----------
        effect_size : float
            Magnitude of the group-mean shift (same scale as :meth:`generate`).
        correlation : float
            Uniform off-diagonal correlation among the shortcut dimensions
            (compound-symmetry structure).  Must be in ``(-1/(k-1), 1)``
            for the covariance matrix to be positive definite.

        Returns
        -------
        SyntheticResult
            Named container with ``embeddings``, ``labels``,
            ``group_labels``, and ``shortcut_dims``.
        """
        if effect_size < 0.0:
            raise ValueError("effect_size must be >= 0")
        k = self.shortcut_dims
        if k > 1:
            min_corr = -1.0 / (k - 1)
            if not (min_corr < correlation < 1.0):
                raise ValueError(
                    f"correlation must be in ({min_corr:.3f}, 1) for k={k} shortcut dims"
                )

        rng = np.random.RandomState(self.seed)

        group_labels = (rng.rand(self.n_samples) > self.group_ratio).astype(np.int64)
        labels = group_labels.copy()

        # Base embeddings: standard-normal noise.
        embeddings = rng.randn(self.n_samples, self.embedding_dim).astype(np.float32)

        true_dims = list(range(k))

        if effect_size > 0.0 and k > 0:
            # Compound-symmetry covariance: diag=1, off-diag=correlation.
            cov = np.full((k, k), correlation, dtype=np.float64)
            np.fill_diagonal(cov, 1.0)
            L = np.linalg.cholesky(cov)  # lower Cholesky factor

            for group_id in (0, 1):
                mask = labels == group_id
                n_group = int(mask.sum())
                mean_shift = effect_size if group_id == 1 else -effect_size
                # Sample z ~ N(0, I_k), then rotate to get N(0, cov).
                z = rng.randn(n_group, k)
                correlated = (z @ L.T + mean_shift).astype(np.float32)
                embeddings[np.ix_(np.where(mask)[0], true_dims)] = correlated

        return SyntheticResult(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            shortcut_dims=true_dims,
        )

    def generate_distributed(self, effect_size: float = 0.8) -> SyntheticResult:
        """Generate a harder synthetic dataset with a distributed shortcut.

        The shortcut signal is spread uniformly across **all** embedding
        dimensions instead of being concentrated in ``shortcut_dims``
        leading dimensions.  Each dimension is shifted by
        ``±effect_size * sqrt(shortcut_dims / embedding_dim)``, which
        preserves the total squared signal energy relative to
        :meth:`generate` while reducing the per-dimension effect size.
        This challenges methods that rely on detecting large shifts in
        individual dimensions.

        Parameters
        ----------
        effect_size : float
            Global effect magnitude.  The per-dimension shift is
            ``effect_size * sqrt(shortcut_dims / embedding_dim)``.

        Returns
        -------
        SyntheticResult
            Named container.  ``shortcut_dims`` lists all embedding
            dimensions because the signal is present in all of them.
        """
        if effect_size < 0.0:
            raise ValueError("effect_size must be >= 0")

        rng = np.random.RandomState(self.seed)

        group_labels = (rng.rand(self.n_samples) > self.group_ratio).astype(np.int64)
        labels = group_labels.copy()

        embeddings = rng.randn(self.n_samples, self.embedding_dim).astype(np.float32)

        if effect_size > 0.0 and self.shortcut_dims > 0:
            per_dim_effect = float(effect_size * np.sqrt(self.shortcut_dims / self.embedding_dim))
            shift = np.where(labels == 1, per_dim_effect, -per_dim_effect).astype(np.float32)
            embeddings += shift[:, np.newaxis]

        # All dims carry the (distributed) shortcut signal.
        true_dims = list(range(self.embedding_dim))

        return SyntheticResult(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            shortcut_dims=true_dims,
        )


def generate_parametric(
    effect_size: float = 0.8,
    n_samples: int = 1000,
    embedding_dim: int = 128,
    shortcut_dims: int = 5,
    group_ratio: float = 0.5,
    seed: int = 42,
) -> SyntheticResult:
    """Convenience function wrapping :class:`SyntheticGenerator`.

    Parameters
    ----------
    effect_size : float
        Shortcut effect magnitude (0.0 = none, 2.0 = very strong).
    n_samples : int
        Number of samples.
    embedding_dim : int
        Total embedding dimensionality.
    shortcut_dims : int
        Number of dimensions carrying the shortcut signal.
    group_ratio : float
        Fraction of samples in the majority group (0.5 = balanced).
    seed : int
        Random seed.

    Returns
    -------
    SyntheticResult
        Named container with ``embeddings``, ``labels``,
        ``group_labels``, and ``shortcut_dims``.
    """
    gen = SyntheticGenerator(
        n_samples=n_samples,
        embedding_dim=embedding_dim,
        shortcut_dims=shortcut_dims,
        group_ratio=group_ratio,
        seed=seed,
    )
    return gen.generate(effect_size=effect_size)


def generate_correlated_parametric(
    effect_size: float = 0.8,
    correlation: float = 0.7,
    n_samples: int = 1000,
    embedding_dim: int = 128,
    shortcut_dims: int = 5,
    group_ratio: float = 0.5,
    seed: int = 42,
) -> SyntheticResult:
    """Convenience function for the correlated-shortcut harder variant.

    Parameters
    ----------
    effect_size : float
        Group-mean shift magnitude.
    correlation : float
        Uniform off-diagonal correlation among shortcut dimensions.
    n_samples, embedding_dim, shortcut_dims, group_ratio, seed :
        Same as :func:`generate_parametric`.

    Returns
    -------
    SyntheticResult
    """
    gen = SyntheticGenerator(
        n_samples=n_samples,
        embedding_dim=embedding_dim,
        shortcut_dims=shortcut_dims,
        group_ratio=group_ratio,
        seed=seed,
    )
    return gen.generate_correlated(effect_size=effect_size, correlation=correlation)


def generate_distributed_parametric(
    effect_size: float = 0.8,
    n_samples: int = 1000,
    embedding_dim: int = 128,
    shortcut_dims: int = 5,
    group_ratio: float = 0.5,
    seed: int = 42,
) -> SyntheticResult:
    """Convenience function for the distributed-shortcut harder variant.

    Parameters
    ----------
    effect_size : float
        Global effect magnitude; per-dim shift =
        ``effect_size * sqrt(shortcut_dims / embedding_dim)``.
    n_samples, embedding_dim, shortcut_dims, group_ratio, seed :
        Same as :func:`generate_parametric`.

    Returns
    -------
    SyntheticResult
    """
    gen = SyntheticGenerator(
        n_samples=n_samples,
        embedding_dim=embedding_dim,
        shortcut_dims=shortcut_dims,
        group_ratio=group_ratio,
        seed=seed,
    )
    return gen.generate_distributed(effect_size=effect_size)
