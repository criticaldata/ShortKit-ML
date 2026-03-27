"""Tests for VAE detector."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from shortcut_detect.vae import VAEDetector, vae_arch
from shortcut_detect.vae.latent_analyzer import (
    compute_mpwd_per_dimension,
    compute_predictiveness_per_dimension,
    rank_candidate_dimensions,
)


def _make_synthetic_images(n: int = 50, img_size: int = 32, channels: int = 3, seed: int = 0):
    """Generate small synthetic images for fast testing."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, size=(n, channels, img_size, img_size)).astype(np.float32)
    return torch.from_numpy(x)


def test_latent_analyzer_mpwd():
    """MPWD returns one value per dimension."""
    rng = np.random.default_rng(1)
    latents = rng.normal(0, 1, size=(100, 5))
    labels = np.array([0] * 50 + [1] * 50)

    mpwd = compute_mpwd_per_dimension(latents, labels, latent_dim=5, num_classes=2)
    assert mpwd.shape == (5,)
    assert np.all(mpwd >= 0)


def test_latent_analyzer_predictiveness():
    """Predictiveness extracts classifier weights."""
    fc = torch.nn.Linear(5, 2)
    torch.nn.init.ones_(fc.weight)

    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = fc

    pred = compute_predictiveness_per_dimension(FakeClassifier(), latent_dim=5)
    assert pred.shape == (5,)
    assert np.all(pred >= 0)


def test_rank_candidate_dimensions():
    """Rank and flag dimensions above threshold."""
    predictiveness = np.array([0.1, 0.9, 0.3, 0.7, 0.2])
    mpwd = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    sorted_idx, flagged = rank_candidate_dimensions(
        predictiveness, mpwd, predictiveness_threshold=0.5
    )
    assert len(sorted_idx) == 5
    assert sorted_idx[0] == 1  # highest predictiveness
    assert 1 in flagged
    assert 3 in flagged
    assert 0 not in flagged


def test_vae_detector_fit_from_arrays():
    """VAEDetector fits on images and labels."""
    images = _make_synthetic_images(n=60, img_size=32, seed=42)
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)

    detector = VAEDetector(
        latent_dim=4,
        kld_weight=1.0,
        epochs=2,
        classifier_epochs=2,
        batch_size=8,
        random_state=42,
    )
    detector.fit(
        images=images,
        labels=labels,
        img_size=32,
        channels=3,
        num_classes=2,
    )

    report = detector.get_report()
    assert report["method"] == "vae"
    assert report["shortcut_detected"] is not None
    assert report["risk_level"] in {"low", "moderate", "high"}
    assert "per_dimension" in report["report"]
    assert len(report["report"]["per_dimension"]) == 4
    assert report["metrics"]["latent_dim"] == 4


def test_vae_detector_requires_min_samples():
    """VAEDetector raises on too few samples."""
    images = _make_synthetic_images(n=5, img_size=32)
    labels = np.array([0, 0, 1, 1, 1])

    detector = VAEDetector(latent_dim=4, epochs=2, classifier_epochs=2)
    with pytest.raises(ValueError, match="at least 10"):
        detector.fit(
            images=images,
            labels=labels,
            img_size=32,
            channels=3,
            num_classes=2,
        )


def test_vae_detector_fit_requires_images_or_dataloaders():
    """VAEDetector raises when neither images nor dataloaders provided."""
    detector = VAEDetector(latent_dim=4, epochs=2, classifier_epochs=2)
    with pytest.raises(ValueError, match="images and labels"):
        detector.fit(
            img_size=32,
            channels=3,
            num_classes=2,
        )


def test_resnet_vae_encode_supports_grayscale(monkeypatch):
    """Grayscale input is expanded to 3 channels before ResNet encoder."""

    class DummyResnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2048, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Identity()

    monkeypatch.setattr(vae_arch, "resnet50", lambda weights=None: DummyResnet())
    monkeypatch.setattr(vae_arch, "ResNet50_Weights", None)

    model = vae_arch.ResnetVAE(
        input_size=32,
        latent_dim=4,
        input_channels=1,
        num_classes=2,
    )
    x = torch.rand(2, 1, 32, 32)
    mu, log_var = model.encode(x)

    assert mu.shape == (2, 4)
    assert log_var.shape == (2, 4)
