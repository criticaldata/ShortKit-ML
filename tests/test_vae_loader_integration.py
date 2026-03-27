"""Unified loader-mode integration tests for VAE."""

import numpy as np
import pytest
import torch

from shortcut_detect import ShortcutDetector


def test_vae_loader_integration():
    """VAE runs via fit_from_loaders with images array."""
    rng = np.random.default_rng(11)
    n = 60
    img_size = 32
    channels = 3

    images = rng.uniform(0.1, 0.9, size=(n, channels, img_size, img_size)).astype(np.float32)
    images_t = torch.from_numpy(images)
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)

    loader = {
        "images": images_t,
        "labels": labels,
        "img_size": img_size,
        "channels": channels,
        "num_classes": 2,
    }

    detector = ShortcutDetector(
        methods=["vae"],
        vae_latent_dim=4,
        vae_epochs=2,
        vae_classifier_epochs=2,
        vae_batch_size=8,
    )
    detector.fit_from_loaders({"vae": loader})

    result = detector.get_results().get("vae")
    assert result is not None
    assert result["success"] is True
    assert "latent_dim" in result["metrics"]
    assert result["metrics"]["latent_dim"] == 4
    assert "per_dimension" in result["report"]
    assert result["summary_title"] == "VAE (Variational Autoencoder) Shortcut Detection"


def test_vae_run_raises_without_loader():
    """run() raises ValueError directing user to fit_from_loaders."""
    from shortcut_detect.unified import DetectorFactory

    factory = DetectorFactory(seed=42)
    builder = factory.create("vae")

    import numpy as np

    with pytest.raises(ValueError, match="fit_from_loaders"):
        builder.run(
            embeddings=np.zeros((10, 8)),
            labels=np.zeros(10),
            group_labels=np.zeros(10),
            feature_names=None,
            protected_labels=None,
        )
