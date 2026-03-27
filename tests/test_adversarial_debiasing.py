"""Tests for Adversarial Debiasing (M04)."""

import numpy as np
import pytest

from shortcut_detect.datasets import generate_linear_shortcut
from shortcut_detect.mitigation import AdversarialDebiasing
from shortcut_detect.probes import SKLearnProbe


def test_adversarial_debiasing_init():
    """AdversarialDebiasing accepts valid parameters."""
    AdversarialDebiasing(hidden_dim=32, n_epochs=10, random_state=42)
    AdversarialDebiasing(adversary_weight=0.3, batch_size=32)


def test_fit_requires_same_length():
    """fit raises when embeddings and protected_labels length mismatch."""
    debiaser = AdversarialDebiasing(n_epochs=2, random_state=42)
    emb = np.random.randn(10, 8).astype(np.float32)
    labels = np.array([0, 1] * 5)  # 10
    with pytest.raises(ValueError, match="same length"):
        debiaser.fit(emb, labels[:5])


def test_fit_embeddings_2d():
    """fit raises when embeddings not 2D."""
    debiaser = AdversarialDebiasing(n_epochs=2, random_state=42)
    with pytest.raises(ValueError, match="2D"):
        debiaser.fit(np.random.randn(10, 8, 4), np.zeros(10))


def test_transform_before_fit():
    """transform raises when not fitted."""
    debiaser = AdversarialDebiasing(n_epochs=2, random_state=42)
    with pytest.raises(ValueError, match="not been fitted"):
        debiaser.transform(np.random.randn(5, 8))


def test_transform_output_shape():
    """transform returns debiased embeddings with correct shape (n_samples, hidden_dim)."""
    rng = np.random.default_rng(42)
    n, d, h = 50, 16, 8
    emb = rng.standard_normal(size=(n, d)).astype(np.float32)
    protected = rng.integers(0, 2, size=n)

    debiaser = AdversarialDebiasing(
        hidden_dim=h,
        n_epochs=5,
        batch_size=16,
        random_state=42,
    )
    debiaser.fit(emb, protected)
    out = debiaser.transform(emb)

    assert out.shape == (n, h)
    assert out.dtype == np.float64


def test_reproducibility():
    """Same random_state produces same transform output."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(60, 12)).astype(np.float32)
    protected = rng.integers(0, 2, size=60)

    d1 = AdversarialDebiasing(n_epochs=3, batch_size=16, random_state=123)
    d1.fit(emb, protected)
    out1 = d1.transform(emb)

    d2 = AdversarialDebiasing(n_epochs=3, batch_size=16, random_state=123)
    d2.fit(emb, protected)
    out2 = d2.transform(emb)

    np.testing.assert_allclose(out1, out2)


def test_probe_accuracy_drops_after_debiasing():
    """Probe accuracy on protected attribute drops after adversarial debiasing."""
    embeddings, protected_labels = generate_linear_shortcut(
        n_samples=400,
        embedding_dim=24,
        shortcut_dims=4,
        seed=42,
    )
    embeddings = embeddings.astype(np.float32)
    protected_labels = protected_labels.astype(np.int64)

    probe = SKLearnProbe(threshold=0.6, random_state=42)
    probe.fit(embeddings, protected_labels)
    acc_before = probe.metric_value_

    debiaser = AdversarialDebiasing(
        hidden_dim=8,  # Bottleneck forces encoder to discard info
        adversary_weight=1.0,
        n_epochs=100,
        batch_size=32,
        lr=1e-2,
        random_state=42,
    )
    debiaser.fit(embeddings, protected_labels)
    debiased = debiaser.transform(embeddings)

    probe.fit(debiased, protected_labels)
    acc_after = probe.metric_value_

    assert (
        acc_after < acc_before
    ), f"Probe accuracy should drop after debiasing: before={acc_before:.3f}, after={acc_after:.3f}"


def test_fit_with_task_labels():
    """fit accepts task_labels and runs without error."""
    rng = np.random.default_rng(42)
    n, d = 60, 12
    emb = rng.standard_normal(size=(n, d)).astype(np.float32)
    protected = rng.integers(0, 2, size=n)
    task = rng.integers(0, 3, size=n)

    debiaser = AdversarialDebiasing(hidden_dim=8, n_epochs=3, random_state=42)
    debiaser.fit(emb, protected, task_labels=task)
    out = debiaser.transform(emb)

    assert out.shape[0] == n
    assert out.shape[1] == 8


def test_fit_transform():
    """fit_transform returns debiased embeddings."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(40, 10)).astype(np.float32)
    protected = rng.integers(0, 2, size=40)

    debiaser = AdversarialDebiasing(hidden_dim=8, n_epochs=3, random_state=42)
    out = debiaser.fit_transform(emb, protected)

    assert out.shape == (40, 8)
    assert debiaser._fitted


def test_transform_embed_dim_mismatch():
    """transform raises when embed_dim does not match fitted."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(20, 10)).astype(np.float32)
    protected = rng.integers(0, 2, size=20)

    debiaser = AdversarialDebiasing(n_epochs=2, random_state=42)
    debiaser.fit(emb, protected)

    wrong_emb = rng.standard_normal(size=(20, 15)).astype(np.float32)
    with pytest.raises(ValueError, match="embed_dim"):
        debiaser.transform(wrong_emb)


def test_task_labels_length_mismatch():
    """fit raises when task_labels length mismatch."""
    debiaser = AdversarialDebiasing(n_epochs=2, random_state=42)
    emb = np.random.randn(10, 8).astype(np.float32)
    protected = np.zeros(10)
    task = np.zeros(5)
    with pytest.raises(ValueError, match="same length"):
        debiaser.fit(emb, protected, task_labels=task)


def test_fit_supports_string_labels():
    """fit accepts categorical string protected/task labels."""
    rng = np.random.default_rng(42)
    n, d = 40, 10
    emb = rng.standard_normal(size=(n, d)).astype(np.float32)
    protected = np.array(["male" if i % 2 == 0 else "female" for i in range(n)], dtype=object)
    task = np.array(["disease_a" if i % 3 == 0 else "disease_b" for i in range(n)], dtype=object)

    debiaser = AdversarialDebiasing(hidden_dim=8, n_epochs=2, batch_size=16, random_state=42)
    debiaser.fit(emb, protected, task_labels=task)
    out = debiaser.transform(emb)

    assert out.shape == (n, 8)
