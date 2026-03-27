"""Tests for the parametric synthetic data generator."""

from __future__ import annotations

import numpy as np
import pytest

from shortcut_detect.benchmark.synthetic_generator import (
    SyntheticGenerator,
    SyntheticResult,
    generate_correlated_parametric,
    generate_distributed_parametric,
    generate_parametric,
)

# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


class TestOutputShapes:
    @pytest.mark.parametrize(
        "n_samples,embedding_dim,shortcut_dims",
        [
            (100, 32, 3),
            (500, 128, 5),
            (1000, 256, 10),
        ],
    )
    def test_shapes(self, n_samples, embedding_dim, shortcut_dims):
        result = generate_parametric(
            effect_size=0.8,
            n_samples=n_samples,
            embedding_dim=embedding_dim,
            shortcut_dims=shortcut_dims,
        )
        assert isinstance(result, SyntheticResult)
        assert result.embeddings.shape == (n_samples, embedding_dim)
        assert result.labels.shape == (n_samples,)
        assert result.group_labels.shape == (n_samples,)
        assert len(result.shortcut_dims) == shortcut_dims
        assert result.shortcut_dims == list(range(shortcut_dims))

    def test_label_values_are_binary(self):
        result = generate_parametric(effect_size=1.0, n_samples=500)
        assert set(np.unique(result.labels)).issubset({0, 1})
        assert set(np.unique(result.group_labels)).issubset({0, 1})

    def test_embeddings_dtype(self):
        result = generate_parametric(effect_size=0.5, n_samples=200)
        assert result.embeddings.dtype == np.float32


# ---------------------------------------------------------------------------
# Effect size = 0.0  ->  no group separation in shortcut dims
# ---------------------------------------------------------------------------


class TestEffectSizeZero:
    def test_no_separation(self):
        result = generate_parametric(
            effect_size=0.0, n_samples=2000, embedding_dim=64, shortcut_dims=5, seed=0
        )
        # For each shortcut dim, the mean difference between groups should
        # be close to zero (within noise).
        for dim in result.shortcut_dims:
            g0 = result.embeddings[result.group_labels == 0, dim]
            g1 = result.embeddings[result.group_labels == 1, dim]
            diff = abs(g0.mean() - g1.mean())
            assert (
                diff < 0.3
            ), f"Shortcut dim {dim}: mean diff {diff:.3f} too large for effect_size=0"

    def test_shortcut_dims_similar_to_noise_dims(self):
        result = generate_parametric(
            effect_size=0.0, n_samples=2000, embedding_dim=64, shortcut_dims=5, seed=1
        )
        # Variance of shortcut dims should be similar to non-shortcut dims.
        sc_var = result.embeddings[:, result.shortcut_dims].var()
        noise_var = result.embeddings[:, 10:20].var()
        assert abs(sc_var - noise_var) < 0.3


# ---------------------------------------------------------------------------
# Effect size = 2.0  ->  strong group separation in shortcut dims
# ---------------------------------------------------------------------------


class TestEffectSizeStrong:
    def test_strong_separation(self):
        result = generate_parametric(
            effect_size=2.0, n_samples=2000, embedding_dim=64, shortcut_dims=5, seed=0
        )
        for dim in result.shortcut_dims:
            g0 = result.embeddings[result.group_labels == 0, dim]
            g1 = result.embeddings[result.group_labels == 1, dim]
            diff = g1.mean() - g0.mean()
            # Expected separation is 4.0 (shift of +2 for group 1, -2 for group 0).
            assert (
                diff > 3.0
            ), f"Shortcut dim {dim}: mean diff {diff:.3f} too small for effect_size=2.0"

    def test_non_shortcut_dims_unaffected(self):
        result = generate_parametric(
            effect_size=2.0, n_samples=2000, embedding_dim=64, shortcut_dims=5, seed=0
        )
        for dim in range(10, 20):
            g0 = result.embeddings[result.group_labels == 0, dim]
            g1 = result.embeddings[result.group_labels == 1, dim]
            diff = abs(g0.mean() - g1.mean())
            assert diff < 0.3, f"Non-shortcut dim {dim}: mean diff {diff:.3f} should be near zero"


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_output(self):
        r1 = generate_parametric(effect_size=0.8, seed=99)
        r2 = generate_parametric(effect_size=0.8, seed=99)
        np.testing.assert_array_equal(r1.embeddings, r2.embeddings)
        np.testing.assert_array_equal(r1.labels, r2.labels)
        np.testing.assert_array_equal(r1.group_labels, r2.group_labels)
        assert r1.shortcut_dims == r2.shortcut_dims

    def test_different_seed_different_output(self):
        r1 = generate_parametric(effect_size=0.8, seed=1)
        r2 = generate_parametric(effect_size=0.8, seed=2)
        assert not np.array_equal(r1.embeddings, r2.embeddings)

    def test_class_api_reproducibility(self):
        gen = SyntheticGenerator(seed=42)
        r1 = gen.generate(effect_size=1.0)
        r2 = gen.generate(effect_size=1.0)
        np.testing.assert_array_equal(r1.embeddings, r2.embeddings)


# ---------------------------------------------------------------------------
# Group ratio / imbalance
# ---------------------------------------------------------------------------


class TestGroupRatio:
    def test_balanced(self):
        result = generate_parametric(effect_size=0.5, n_samples=10000, group_ratio=0.5, seed=0)
        frac_group0 = (result.group_labels == 0).mean()
        assert 0.45 < frac_group0 < 0.55

    def test_imbalanced_07(self):
        result = generate_parametric(effect_size=0.5, n_samples=10000, group_ratio=0.7, seed=0)
        frac_group0 = (result.group_labels == 0).mean()
        assert 0.65 < frac_group0 < 0.75

    def test_imbalanced_09(self):
        result = generate_parametric(effect_size=0.5, n_samples=10000, group_ratio=0.9, seed=0)
        frac_group0 = (result.group_labels == 0).mean()
        assert 0.85 < frac_group0 < 0.95


# ---------------------------------------------------------------------------
# Shortcut dims correctness
# ---------------------------------------------------------------------------


class TestShortcutDims:
    def test_dims_are_leading(self):
        result = generate_parametric(shortcut_dims=7, embedding_dim=64)
        assert result.shortcut_dims == [0, 1, 2, 3, 4, 5, 6]

    def test_zero_shortcut_dims(self):
        result = generate_parametric(effect_size=1.0, shortcut_dims=0, embedding_dim=32)
        assert result.shortcut_dims == []
        # All dims should be pure noise -- no separation.
        for dim in range(32):
            g0 = result.embeddings[result.group_labels == 0, dim]
            g1 = result.embeddings[result.group_labels == 1, dim]
            diff = abs(g0.mean() - g1.mean())
            assert diff < 0.5


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_effect_size_raises(self):
        with pytest.raises(ValueError, match="effect_size"):
            generate_parametric(effect_size=-0.5)

    def test_invalid_group_ratio_raises(self):
        with pytest.raises(ValueError, match="group_ratio"):
            SyntheticGenerator(group_ratio=0.0)
        with pytest.raises(ValueError, match="group_ratio"):
            SyntheticGenerator(group_ratio=1.0)

    def test_shortcut_dims_exceeds_embedding_dim_raises(self):
        with pytest.raises(ValueError, match="shortcut_dims"):
            SyntheticGenerator(embedding_dim=10, shortcut_dims=11)

    def test_labels_equal_group_labels(self):
        """Labels should be identical to group labels (direct correlation)."""
        result = generate_parametric(effect_size=1.0, n_samples=500, seed=7)
        np.testing.assert_array_equal(result.labels, result.group_labels)


# ---------------------------------------------------------------------------
# Correlated shortcut variant
# ---------------------------------------------------------------------------


class TestGenerateCorrelated:
    def test_output_shapes(self):
        result = generate_correlated_parametric(
            effect_size=0.8, correlation=0.7, n_samples=300, embedding_dim=32, shortcut_dims=4
        )
        assert result.embeddings.shape == (300, 32)
        assert result.labels.shape == (300,)
        assert result.shortcut_dims == [0, 1, 2, 3]

    def test_dtype(self):
        result = generate_correlated_parametric(effect_size=0.8, n_samples=200)
        assert result.embeddings.dtype == np.float32

    def test_within_group_correlation(self):
        result = generate_correlated_parametric(
            effect_size=0.8,
            correlation=0.7,
            n_samples=3000,
            embedding_dim=32,
            shortcut_dims=4,
            seed=0,
        )
        for g in [0, 1]:
            block = result.embeddings[result.labels == g, :4]
            corr = np.corrcoef(block.T)
            off_diag_mean = corr[np.triu_indices(4, k=1)].mean()
            assert (
                0.55 < off_diag_mean < 0.85
            ), f"Group {g}: expected within-group correlation ~0.7, got {off_diag_mean:.3f}"

    def test_group_separation_in_shortcut_dims(self):
        result = generate_correlated_parametric(
            effect_size=1.5,
            correlation=0.5,
            n_samples=2000,
            embedding_dim=32,
            shortcut_dims=4,
            seed=1,
        )
        for dim in result.shortcut_dims:
            g0 = result.embeddings[result.labels == 0, dim]
            g1 = result.embeddings[result.labels == 1, dim]
            diff = g1.mean() - g0.mean()
            assert diff > 1.5, f"Dim {dim}: mean diff {diff:.3f} too small"

    def test_non_shortcut_dims_unaffected(self):
        result = generate_correlated_parametric(
            effect_size=1.5,
            correlation=0.5,
            n_samples=2000,
            embedding_dim=32,
            shortcut_dims=4,
            seed=1,
        )
        for dim in range(8, 20):
            g0 = result.embeddings[result.labels == 0, dim]
            g1 = result.embeddings[result.labels == 1, dim]
            assert abs(g0.mean() - g1.mean()) < 0.3

    def test_reproducibility(self):
        r1 = generate_correlated_parametric(effect_size=0.8, seed=7)
        r2 = generate_correlated_parametric(effect_size=0.8, seed=7)
        np.testing.assert_array_equal(r1.embeddings, r2.embeddings)

    def test_invalid_effect_size_raises(self):
        with pytest.raises(ValueError, match="effect_size"):
            gen = SyntheticGenerator()
            gen.generate_correlated(effect_size=-1.0)

    def test_invalid_correlation_raises(self):
        with pytest.raises(ValueError, match="correlation"):
            gen = SyntheticGenerator(shortcut_dims=4)
            gen.generate_correlated(effect_size=0.8, correlation=-0.5)

    def test_zero_effect_size_no_separation(self):
        result = generate_correlated_parametric(
            effect_size=0.0,
            correlation=0.7,
            n_samples=2000,
            embedding_dim=32,
            shortcut_dims=4,
            seed=0,
        )
        for dim in result.shortcut_dims:
            g0 = result.embeddings[result.labels == 0, dim]
            g1 = result.embeddings[result.labels == 1, dim]
            assert abs(g0.mean() - g1.mean()) < 0.3


# ---------------------------------------------------------------------------
# Distributed shortcut variant
# ---------------------------------------------------------------------------


class TestGenerateDistributed:
    def test_output_shapes(self):
        result = generate_distributed_parametric(
            effect_size=0.8, n_samples=300, embedding_dim=32, shortcut_dims=4
        )
        assert result.embeddings.shape == (300, 32)
        # All dims carry the signal.
        assert result.shortcut_dims == list(range(32))

    def test_dtype(self):
        result = generate_distributed_parametric(effect_size=0.8, n_samples=200)
        assert result.embeddings.dtype == np.float32

    def test_per_dim_effect_size(self):
        k, d = 5, 128
        effect = 0.8
        result = generate_distributed_parametric(
            effect_size=effect, n_samples=5000, embedding_dim=d, shortcut_dims=k, seed=0
        )
        expected_per_dim_shift = 2 * effect * np.sqrt(k / d)
        shifts = []
        for dim in range(d):
            g0 = result.embeddings[result.labels == 0, dim]
            g1 = result.embeddings[result.labels == 1, dim]
            shifts.append(g1.mean() - g0.mean())
        mean_shift = np.mean(shifts)
        assert (
            abs(mean_shift - expected_per_dim_shift) < 0.05
        ), f"Expected per-dim shift ~{expected_per_dim_shift:.3f}, got {mean_shift:.3f}"

    def test_all_dims_shifted(self):
        result = generate_distributed_parametric(
            effect_size=2.0, n_samples=3000, embedding_dim=16, shortcut_dims=4, seed=1
        )
        for dim in range(16):
            g0 = result.embeddings[result.labels == 0, dim]
            g1 = result.embeddings[result.labels == 1, dim]
            shift = g1.mean() - g0.mean()
            assert shift > 0.5, f"Dim {dim}: shift {shift:.3f} too small"

    def test_reproducibility(self):
        r1 = generate_distributed_parametric(effect_size=0.8, seed=99)
        r2 = generate_distributed_parametric(effect_size=0.8, seed=99)
        np.testing.assert_array_equal(r1.embeddings, r2.embeddings)

    def test_zero_effect_size_no_separation(self):
        result = generate_distributed_parametric(
            effect_size=0.0, n_samples=2000, embedding_dim=32, shortcut_dims=5, seed=0
        )
        for dim in range(32):
            g0 = result.embeddings[result.labels == 0, dim]
            g1 = result.embeddings[result.labels == 1, dim]
            assert abs(g0.mean() - g1.mean()) < 0.3

    def test_invalid_effect_size_raises(self):
        with pytest.raises(ValueError, match="effect_size"):
            gen = SyntheticGenerator()
            gen.generate_distributed(effect_size=-0.1)


# ---------------------------------------------------------------------------
# Import from benchmark package
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_from_benchmark(self):
        from shortcut_detect.benchmark import (
            SyntheticGenerator,
            SyntheticResult,
            generate_correlated_parametric,
            generate_distributed_parametric,
            generate_parametric,
        )

        assert SyntheticGenerator is not None
        assert SyntheticResult is not None
        assert generate_parametric is not None
        assert generate_correlated_parametric is not None
        assert generate_distributed_parametric is not None
