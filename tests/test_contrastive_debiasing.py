"""Tests for Contrastive Debiasing (M07)."""

import numpy as np
import pytest

from shortcut_detect.mitigation import ContrastiveDebiasing
from shortcut_detect.probes import SKLearnProbe


def test_contrastive_debiasing_init():
    """ContrastiveDebiasing accepts valid parameters."""
    ContrastiveDebiasing(hidden_dim=32, n_epochs=10, random_state=42)
    ContrastiveDebiasing(temperature=0.1, contrastive_weight=0.5, batch_size=32)


def test_fit_requires_task_and_group():
    """fit raises when task_labels or group_labels length mismatch."""
    cnc = ContrastiveDebiasing(n_epochs=2, random_state=42)
    emb = np.random.randn(20, 8).astype(np.float32)
    task = np.array([0, 1] * 10)
    group = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="same length"):
        cnc.fit(emb, task[:10], group)
    with pytest.raises(ValueError, match="same length"):
        cnc.fit(emb, task, group[:10])


def test_fit_embeddings_2d():
    """fit raises when embeddings not 2D."""
    cnc = ContrastiveDebiasing(n_epochs=2, random_state=42)
    task = np.array([0, 1] * 5)
    group = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="2D"):
        cnc.fit(np.random.randn(10, 8, 4), task, group)


def test_fit_requires_at_least_2_groups():
    """fit raises when only one group present."""
    cnc = ContrastiveDebiasing(n_epochs=2, random_state=42)
    emb = np.random.randn(20, 8).astype(np.float32)
    task = np.array([0, 1] * 10)
    group = np.zeros(20)
    with pytest.raises(ValueError, match="at least 2 groups"):
        cnc.fit(emb, task, group)


def test_fit_requires_at_least_2_task_classes():
    """fit raises when only one task class present."""
    cnc = ContrastiveDebiasing(n_epochs=2, random_state=42)
    emb = np.random.randn(20, 8).astype(np.float32)
    task = np.zeros(20)
    group = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="at least 2 task classes"):
        cnc.fit(emb, task, group)


def test_transform_before_fit():
    """transform raises when not fitted."""
    cnc = ContrastiveDebiasing(n_epochs=2, random_state=42)
    with pytest.raises(ValueError, match="not been fitted"):
        cnc.transform(np.random.randn(5, 8))


def test_transform_output_shape():
    """transform returns debiased embeddings with correct shape (n_samples, hidden_dim)."""
    rng = np.random.default_rng(42)
    n, d, h = 80, 16, 8
    emb = rng.standard_normal(size=(n, d)).astype(np.float32)
    task = rng.integers(0, 2, size=n)
    group = rng.integers(0, 2, size=n)

    cnc = ContrastiveDebiasing(
        hidden_dim=h,
        n_epochs=5,
        batch_size=16,
        random_state=42,
    )
    cnc.fit(emb, task, group)
    out = cnc.transform(emb)

    assert out.shape == (n, h)
    assert out.dtype == np.float64


def test_reproducibility():
    """Same random_state produces same transform output."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(80, 12)).astype(np.float32)
    task = rng.integers(0, 2, size=80)
    group = rng.integers(0, 2, size=80)

    c1 = ContrastiveDebiasing(n_epochs=3, batch_size=16, random_state=123)
    c1.fit(emb, task, group)
    out1 = c1.transform(emb)

    c2 = ContrastiveDebiasing(n_epochs=3, batch_size=16, random_state=123)
    c2.fit(emb, task, group)
    out2 = c2.transform(emb)

    np.testing.assert_allclose(out1, out2)


def test_probe_accuracy_drops_after_debiasing():
    """Probe accuracy on group attribute drops after contrastive debiasing."""
    # Build embeddings with explicit task (dims 0-2) and group (dims 3-5) encoding
    rng = np.random.default_rng(42)
    n = 400
    d = 24
    embeddings = rng.standard_normal(size=(n, d)).astype(np.float32)
    task_labels = rng.integers(0, 2, size=n)
    group_labels = rng.integers(0, 2, size=n)
    # Task signal in first 3 dims
    for i in range(3):
        embeddings[task_labels == 0, i] -= 2
        embeddings[task_labels == 1, i] += 2
    # Group signal in dims 3-5 (spurious)
    for i in range(3, 6):
        embeddings[group_labels == 0, i] -= 2
        embeddings[group_labels == 1, i] += 2

    probe = SKLearnProbe(threshold=0.6, random_state=42)
    probe.fit(embeddings, group_labels)
    acc_before = probe.metric_value_

    cnc = ContrastiveDebiasing(
        hidden_dim=8,
        temperature=0.05,
        contrastive_weight=0.75,
        n_epochs=100,
        batch_size=32,
        lr=1e-2,
        random_state=42,
    )
    cnc.fit(embeddings, task_labels, group_labels)
    debiased = cnc.transform(embeddings)

    probe.fit(debiased, group_labels)
    acc_after = probe.metric_value_

    assert (
        acc_after < acc_before
    ), f"Probe accuracy should drop after debiasing: before={acc_before:.3f}, after={acc_after:.3f}"


def test_fit_transform():
    """fit_transform returns debiased embeddings."""
    rng = np.random.default_rng(42)
    n, d = 80, 10
    emb = rng.standard_normal(size=(n, d)).astype(np.float32)
    task = rng.integers(0, 2, size=n)
    group = rng.integers(0, 2, size=n)

    cnc = ContrastiveDebiasing(hidden_dim=8, n_epochs=3, random_state=42)
    out = cnc.fit_transform(emb, task, group)

    assert out.shape == (n, 8)
    assert cnc._fitted


def test_transform_embed_dim_mismatch():
    """transform raises when embed_dim does not match fitted."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(40, 10)).astype(np.float32)
    task = rng.integers(0, 2, size=40)
    group = rng.integers(0, 2, size=40)

    cnc = ContrastiveDebiasing(n_epochs=2, random_state=42)
    cnc.fit(emb, task, group)

    wrong_emb = rng.standard_normal(size=(40, 15)).astype(np.float32)
    with pytest.raises(ValueError, match="embed_dim"):
        cnc.transform(wrong_emb)


def test_task_labels_length_mismatch():
    """fit raises when task_labels length mismatch."""
    cnc = ContrastiveDebiasing(n_epochs=2, random_state=42)
    emb = np.random.randn(20, 8).astype(np.float32)
    task = np.zeros(10)
    group = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="same length"):
        cnc.fit(emb, task, group)


def test_fit_supports_string_labels():
    """fit accepts categorical string task/group labels."""
    rng = np.random.default_rng(42)
    n, d = 60, 10
    emb = rng.standard_normal(size=(n, d)).astype(np.float32)
    task = np.array(["class_a" if i % 2 == 0 else "class_b" for i in range(n)], dtype=object)
    group = np.array(["group_1" if i % 3 == 0 else "group_2" for i in range(n)], dtype=object)

    cnc = ContrastiveDebiasing(hidden_dim=8, n_epochs=2, batch_size=16, random_state=42)
    cnc.fit(emb, task, group)
    out = cnc.transform(emb)

    assert out.shape == (n, 8)
