"""Tests for Explanation Regularization (M05 RRR)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from shortcut_detect.mitigation import ExplanationRegularization


def _make_tiny_cnn(in_channels=1, h=32, w=32, n_classes=2):
    """Minimal CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, n_classes),
    )


def test_init():
    """ExplanationRegularization accepts valid parameters."""
    ExplanationRegularization(lambda_rrr=1.0, n_epochs=5, random_state=42)
    ExplanationRegularization(lambda_rrr=0.5, lr=1e-3, batch_size=4)


def test_fit_updates_model():
    """fit updates model parameters in-place."""
    torch.manual_seed(42)
    model = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)
    params_before = [p.clone() for p in model.parameters()]

    n = 16
    images = torch.randn(n, 1, 16, 16)
    labels = np.array([0, 1] * 8)
    masks = np.zeros((n, 16, 16), dtype=np.float32)
    masks[:, 8:, :] = 1.0

    rrr = ExplanationRegularization(
        lambda_rrr=0.1,
        n_epochs=2,
        batch_size=4,
        random_state=42,
    )
    rrr.fit(model, images, labels, masks)

    for p_before, p_after in zip(params_before, model.parameters(), strict=False):
        assert not torch.allclose(p_before, p_after), "Model parameters should change"


def test_labels_length_mismatch():
    """fit raises when labels length does not match images."""
    model = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)
    images = torch.randn(10, 1, 16, 16)
    labels = np.array([0, 1] * 3)
    masks = np.zeros((10, 16, 16), dtype=np.float32)

    rrr = ExplanationRegularization(n_epochs=1, random_state=42)
    with pytest.raises(ValueError, match="labels length"):
        rrr.fit(model, images, labels, masks)


def test_mask_shape_mismatch():
    """fit raises when shortcut_masks batch size does not match images."""
    model = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)
    images = torch.randn(10, 1, 16, 16)
    labels = np.array([0, 1] * 5)
    masks = np.zeros((5, 16, 16), dtype=np.float32)

    rrr = ExplanationRegularization(n_epochs=1, random_state=42)
    with pytest.raises(ValueError, match="shortcut_masks batch size"):
        rrr.fit(model, images, labels, masks)


def test_mask_shape_single_broadcast():
    """fit accepts single mask broadcast to all images."""
    model = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)
    images = torch.randn(6, 1, 16, 16)
    labels = np.array([0, 1] * 3)
    masks = np.zeros((1, 16, 16), dtype=np.float32)
    masks[0, 8:, :] = 1.0

    rrr = ExplanationRegularization(n_epochs=1, batch_size=6, random_state=42)
    rrr.fit(model, images, labels, masks)
    assert len(rrr._history) == 1


def test_history_recorded():
    """fit records training history."""
    model = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)
    images = torch.randn(8, 1, 16, 16)
    labels = np.array([0, 1] * 4)
    masks = np.zeros((8, 16, 16), dtype=np.float32)

    rrr = ExplanationRegularization(n_epochs=3, batch_size=4, random_state=42)
    rrr.fit(model, images, labels, masks)

    assert len(rrr._history) == 3
    for h in rrr._history:
        assert "epoch" in h
        assert "ce_loss" in h
        assert "penalty" in h


def test_reproducibility():
    """Same random_state produces same training outcome."""
    model1 = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)
    model2 = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)
    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        p2.data.copy_(p1.data)

    images = torch.randn(8, 1, 16, 16)
    labels = np.array([0, 1] * 4)
    masks = np.zeros((8, 16, 16), dtype=np.float32)

    rrr = ExplanationRegularization(n_epochs=2, random_state=123)
    rrr.fit(model1, images, labels, masks)
    rrr.fit(model2, images, labels, masks)

    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        torch.testing.assert_close(p1, p2)


def test_model_returns_tuple():
    """fit works when model returns (logits,) tuple."""

    class TupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _make_tiny_cnn(in_channels=1, h=16, w=16, n_classes=2)

        def forward(self, x):
            return (self.net(x),)

    model = TupleModel()
    images = torch.randn(4, 1, 16, 16)
    labels = np.array([0, 1, 0, 1])
    masks = np.zeros((4, 16, 16), dtype=np.float32)

    rrr = ExplanationRegularization(n_epochs=1, head=0, random_state=42)
    rrr.fit(model, images, labels, masks)
    assert len(rrr._history) == 1
