"""Tests for reusable synthetic benchmark generators."""

from __future__ import annotations

import numpy as np
import pytest

from shortcut_detect.benchmark.synthetic import (
    MAX_SHORTCUT_EFFECT_SIZE,
    MIN_SHORTCUT_EFFECT_SIZE,
    generate_parametric_shortcut_dataset,
)


def test_parametric_shortcut_dataset_is_seed_reproducible():
    ds_a = generate_parametric_shortcut_dataset(
        n_samples=128,
        embedding_dim=16,
        shortcut_dims=3,
        effect_size=0.8,
        seed=17,
    )
    ds_b = generate_parametric_shortcut_dataset(
        n_samples=128,
        embedding_dim=16,
        shortcut_dims=3,
        effect_size=0.8,
        seed=17,
    )

    assert np.array_equal(ds_a.embeddings, ds_b.embeddings)
    assert np.array_equal(ds_a.labels, ds_b.labels)
    assert np.array_equal(ds_a.shortcut_dim_labels, ds_b.shortcut_dim_labels)
    assert np.array_equal(ds_a.shortcut_dim_indices, ds_b.shortcut_dim_indices)


def test_parametric_shortcut_dataset_returns_ground_truth_dimension_labels():
    ds = generate_parametric_shortcut_dataset(
        n_samples=64,
        embedding_dim=12,
        shortcut_dims=4,
        effect_size=1.2,
        seed=9,
    )

    assert ds.embeddings.shape == (64, 12)
    assert ds.labels.shape == (64,)
    assert ds.shortcut_dim_labels.shape == (12,)
    assert ds.shortcut_dim_labels.dtype == np.bool_
    assert ds.shortcut_dim_indices.tolist() == [0, 1, 2, 3]
    assert ds.shortcut_dim_labels.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_parametric_shortcut_effect_size_controls_mean_separation():
    subtle = generate_parametric_shortcut_dataset(
        n_samples=5000,
        embedding_dim=10,
        shortcut_dims=2,
        effect_size=0.2,
        seed=123,
    )
    strong = generate_parametric_shortcut_dataset(
        n_samples=5000,
        embedding_dim=10,
        shortcut_dims=2,
        effect_size=2.0,
        seed=123,
    )

    subtle_gap = (
        subtle.embeddings[subtle.labels == 1, 0].mean()
        - subtle.embeddings[subtle.labels == 0, 0].mean()
    )
    strong_gap = (
        strong.embeddings[strong.labels == 1, 0].mean()
        - strong.embeddings[strong.labels == 0, 0].mean()
    )

    assert strong_gap > subtle_gap
    assert strong_gap > 3.0
    assert subtle_gap > 0.1


@pytest.mark.parametrize(
    "effect_size", [MIN_SHORTCUT_EFFECT_SIZE - 0.01, MAX_SHORTCUT_EFFECT_SIZE + 0.01]
)
def test_parametric_shortcut_dataset_validates_effect_size(effect_size: float):
    with pytest.raises(ValueError, match="effect_size must be between"):
        generate_parametric_shortcut_dataset(effect_size=effect_size)
