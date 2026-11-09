"""The shared small CNN and training loop of the image tracks.

Mitigation submissions may use any model, but using these keeps comparisons
about the method rather than the backbone. The image-detection track trains
:func:`make_cnn` with :func:`train_classifier` on each dataset and hands the
fitted model to ``detect``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def make_cnn(n_classes: int = 2) -> nn.Module:
    """Three conv blocks (16, 32, 64 channels, BatchNorm, ReLU), global average pool, linear head."""

    def block(c_in: int, c_out: int) -> list[nn.Module]:
        return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]

    return nn.Sequential(
        *block(3, 16),
        nn.MaxPool2d(2),
        *block(16, 32),
        nn.MaxPool2d(2),
        *block(32, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, n_classes),
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def train_classifier(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 3e-3,
    sample_weight: np.ndarray | None = None,
    sampling_weight: np.ndarray | None = None,
    model: nn.Module | None = None,
) -> nn.Module:
    """Fit a classifier with Adam on cross-entropy. Returns it in eval mode.

    ``sample_weight`` weights each example's loss; ``sampling_weight`` instead
    draws each epoch's examples with replacement in proportion to it (for
    example 1 / group size for group-balanced sampling).
    """
    seed_everything(seed)
    model = model or make_cnn(int(np.max(labels)) + 1)
    X = torch.as_tensor(images, dtype=torch.float32)
    Y = torch.as_tensor(labels, dtype=torch.long)
    W = (
        torch.ones(len(Y))
        if sample_weight is None
        else torch.as_tensor(sample_weight, dtype=torch.float32)
    )
    W = W / W.mean()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        if sampling_weight is None:
            order = torch.randperm(len(Y), generator=g)
        else:
            p = torch.as_tensor(sampling_weight, dtype=torch.float64)
            order = torch.multinomial(p / p.sum(), len(Y), replacement=True, generator=g)
        for start in range(0, len(Y), batch_size):
            idx = order[start : start + batch_size]
            loss = (
                nn.functional.cross_entropy(model(X[idx]), Y[idx], reduction="none") * W[idx]
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    X = torch.as_tensor(images, dtype=torch.float32)
    out = [model(X[i : i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


class TorchPredictor:
    """Wraps a torch model so it satisfies the mitigation tracks' ``predict(images)`` contract."""

    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, images: np.ndarray) -> np.ndarray:
        return predict(self.model, images)
