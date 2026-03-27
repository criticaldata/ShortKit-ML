"""Tests for Last Layer Retraining (M06 DFR)."""

import numpy as np
import pytest

from shortcut_detect.mitigation import LastLayerRetraining


def test_init():
    """LastLayerRetraining accepts valid parameters."""
    LastLayerRetraining(C=1.0, penalty="l1", random_state=42)
    LastLayerRetraining(C=0.1, penalty="l2", class_weight="balanced")


def test_init_invalid_penalty():
    """init raises when penalty is not l1 or l2."""
    with pytest.raises(ValueError, match="penalty must be"):
        LastLayerRetraining(penalty="elasticnet")


def test_fit_requires_same_length():
    """fit raises when embeddings, task_labels, group_labels length mismatch."""
    dfr = LastLayerRetraining(random_state=42)
    emb = np.random.randn(10, 8)
    task = np.array([0, 1] * 5)
    group = np.array([0, 1] * 3)  # 6
    with pytest.raises(ValueError, match="same length"):
        dfr.fit(emb, task, group)


def test_fit_embeddings_2d():
    """fit raises when embeddings not 2D."""
    dfr = LastLayerRetraining(random_state=42)
    with pytest.raises(ValueError, match="2D"):
        dfr.fit(np.random.randn(10, 8, 4), np.zeros(10), np.zeros(10))


def test_predict_before_fit():
    """predict raises when not fitted."""
    dfr = LastLayerRetraining(random_state=42)
    with pytest.raises(ValueError, match="not been fitted"):
        dfr.predict(np.random.randn(5, 8))


def test_fit_predict_output_shape():
    """fit_predict returns predictions with correct shape."""
    rng = np.random.default_rng(42)
    n, d = 50, 16
    emb = rng.standard_normal(size=(n, d))
    task = rng.integers(0, 3, size=n)
    group = rng.integers(0, 2, size=n)

    dfr = LastLayerRetraining(C=1.0, random_state=42)
    preds = dfr.fit_predict(emb, task, group)

    assert preds.shape == (n,)
    assert preds.dtype in (np.int64, np.int32, int)


def test_predict_output_shape():
    """predict returns predictions with correct shape."""
    rng = np.random.default_rng(42)
    n, d = 40, 12
    emb = rng.standard_normal(size=(n, d))
    task = rng.integers(0, 2, size=n)
    group = rng.integers(0, 2, size=n)

    dfr = LastLayerRetraining(random_state=42)
    dfr.fit(emb, task, group)
    preds = dfr.predict(emb)

    assert preds.shape == (n,)


def test_balanced_subset():
    """Balanced subset has correct size (min per group)."""
    rng = np.random.default_rng(42)
    # 3 groups: 20, 15, 10 -> min 10
    n = 45
    emb = rng.standard_normal(size=(n, 8))
    task = rng.integers(0, 2, size=n)  # binary task
    group = np.array([0] * 20 + [1] * 15 + [2] * 10)

    dfr = LastLayerRetraining(random_state=42)
    dfr.fit(emb, task, group)

    assert dfr._n_balanced == 30  # 10 per group
    assert dfr._n_groups == 3


def test_reproducibility():
    """Same random_state produces same predictions."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(60, 12))
    task = rng.integers(0, 3, size=60)
    group = rng.integers(0, 2, size=60)

    d1 = LastLayerRetraining(random_state=123)
    pred1 = d1.fit_predict(emb, task, group)

    d2 = LastLayerRetraining(random_state=123)
    pred2 = d2.fit_predict(emb, task, group)

    np.testing.assert_array_equal(pred1, pred2)


def test_fit_predict():
    """fit_predict returns predictions consistent with fit then predict."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(40, 10))
    task = rng.integers(0, 2, size=40)
    group = rng.integers(0, 2, size=40)

    dfr = LastLayerRetraining(random_state=42)
    preds_fp = dfr.fit_predict(emb, task, group)
    preds_sep = dfr.predict(emb)

    np.testing.assert_array_equal(preds_fp, preds_sep)


def test_single_group():
    """Single group uses all samples in balanced subset."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(20, 8))
    task = rng.integers(0, 2, size=20)
    group = np.zeros(20, dtype=int)  # all same group

    dfr = LastLayerRetraining(random_state=42)
    dfr.fit(emb, task, group)
    preds = dfr.predict(emb)

    assert dfr._n_balanced == 20
    assert dfr._n_groups == 1
    assert preds.shape == (20,)


def test_transform_embed_dim_mismatch():
    """predict raises when embed_dim does not match fitted."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(20, 10))
    task = rng.integers(0, 2, size=20)
    group = rng.integers(0, 2, size=20)

    dfr = LastLayerRetraining(random_state=42)
    dfr.fit(emb, task, group)

    wrong_emb = rng.standard_normal(size=(20, 15))
    with pytest.raises(ValueError, match="embed_dim"):
        dfr.predict(wrong_emb)


def test_scaler_and_classifier_exposed():
    """scaler_ and classifier_ are exposed after fit."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(30, 8))
    task = rng.integers(0, 2, size=30)
    group = rng.integers(0, 2, size=30)

    dfr = LastLayerRetraining(random_state=42)
    assert dfr.scaler_ is None
    assert dfr.classifier_ is None

    dfr.fit(emb, task, group)
    assert dfr.scaler_ is not None
    assert dfr.classifier_ is not None
    assert hasattr(dfr.scaler_, "transform")
    assert hasattr(dfr.classifier_, "predict")


def test_fit_predict_supports_string_labels():
    """fit_predict supports categorical string task/group labels."""
    rng = np.random.default_rng(42)
    n, d = 60, 12
    emb = rng.standard_normal(size=(n, d))
    task = np.array(["neg" if i % 2 == 0 else "pos" for i in range(n)], dtype=object)
    group = np.array(["a" if i % 3 == 0 else "b" for i in range(n)], dtype=object)

    dfr = LastLayerRetraining(random_state=42)
    preds = dfr.fit_predict(emb, task, group)

    assert preds.shape == (n,)
    assert set(np.unique(preds)).issubset({"neg", "pos"})
