"""Validate that SyntheticGenerator produces the intended Cohen's d effect sizes.

For each requested effect_size, we generate data and compute the empirical
Cohen's d on shortcut dimensions.  The generator shifts group means by
+/- effect_size around a standard-normal base, so the expected Cohen's d
is ``2 * effect_size`` (total mean separation divided by the pooled std of ~1).

S04 – Effect size calibration validation (#64)
"""

from __future__ import annotations

import numpy as np
import pytest

from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator


def _empirical_cohens_d(values_g0: np.ndarray, values_g1: np.ndarray) -> float:
    """Compute Cohen's d = |mean_g0 - mean_g1| / pooled_std."""
    n0, n1 = len(values_g0), len(values_g1)
    if n0 < 2 or n1 < 2:
        return 0.0
    var0 = np.var(values_g0, ddof=1)
    var1 = np.var(values_g1, ddof=1)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
    if pooled_std == 0:
        return 0.0
    return float(np.abs(np.mean(values_g0) - np.mean(values_g1)) / pooled_std)


@pytest.mark.parametrize("effect_size", [0.0, 0.2, 0.5, 0.8, 1.2, 2.0])
def test_effect_size_calibration(effect_size: float) -> None:
    """Empirical Cohen's d on shortcut dims should match the intended effect size.

    The generator applies ``+/- effect_size`` shifts to each group on top of
    standard-normal noise, producing an expected Cohen's d of
    ``2 * effect_size``.  We verify the average empirical d across shortcut
    dimensions is within 20% of that target (or < 0.15 for the null case).
    """
    gen = SyntheticGenerator(
        n_samples=5000,
        embedding_dim=64,
        shortcut_dims=5,
        seed=42,
    )
    result = gen.generate(effect_size=effect_size)

    empirical_ds: list[float] = []
    for dim in result.shortcut_dims:
        g0 = result.embeddings[result.group_labels == 0, dim]
        g1 = result.embeddings[result.group_labels == 1, dim]
        empirical_ds.append(_empirical_cohens_d(g0, g1))

    mean_d = float(np.mean(empirical_ds))

    # The expected Cohen's d is 2 * effect_size because the generator
    # shifts group 0 by -effect_size and group 1 by +effect_size.
    expected_d = 2.0 * effect_size

    if effect_size == 0.0:
        # Null case: empirical d should be near zero (just noise).
        assert mean_d < 0.15, f"effect_size=0: expected near-zero Cohen's d, got {mean_d:.4f}"
    else:
        # Non-null: within 20% relative tolerance of the expected d.
        relative_error = abs(mean_d - expected_d) / expected_d
        assert relative_error < 0.20, (
            f"effect_size={effect_size}: expected d~{expected_d:.2f}, "
            f"got {mean_d:.4f} (relative error {relative_error:.1%})"
        )
