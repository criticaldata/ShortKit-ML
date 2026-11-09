"""Saliency helpers for image-detection submissions (used by the baselines)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def input_gradient(model: nn.Module, images: np.ndarray) -> np.ndarray:
    """|d logit of the predicted class / d input|, summed over channels: (n, H, W)."""
    X = torch.as_tensor(images, dtype=torch.float32).requires_grad_(True)
    logits = model(X)
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    return X.grad.abs().sum(1).numpy()


@torch.no_grad()
def occlusion_maps(
    model: nn.Module, images: np.ndarray, size: int = 4, stride: int = 2, fill: float = 0.5
) -> np.ndarray:
    """Drop in predicted-class probability when a ``size`` square is filled with ``fill``: (n, H, W)."""
    X = torch.as_tensor(images, dtype=torch.float32)
    n, _, H, W = X.shape
    base = torch.softmax(model(X), 1)
    cls = base.argmax(1, keepdim=True)
    p0 = base.gather(1, cls).squeeze(1)
    heat = torch.zeros(n, H, W)
    count = torch.zeros(H, W)
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            Xo = X.clone()
            Xo[:, :, r : r + size, c : c + size] = fill
            p = torch.softmax(model(Xo), 1).gather(1, cls).squeeze(1)
            heat[:, r : r + size, c : c + size] += (p0 - p).clamp(min=0)[:, None, None]
            count[r : r + size, c : c + size] += 1
    return (heat / count.clamp(min=1)).numpy()


def top_mass_concentration(maps: np.ndarray, top: float = 0.02) -> float:
    """Share of the dataset-averaged (per-image normalized) map held by its top ``top`` pixels."""
    m = maps / (maps.sum(axis=(1, 2), keepdims=True) + 1e-12)
    avg = m.mean(0).ravel()
    k = max(1, int(top * avg.size))
    return float(np.sort(avg)[-k:].sum())


@torch.no_grad()
def occlusion_flips(
    model: nn.Module, images: np.ndarray, size: int = 4, stride: int = 2, fill: float = 0.5
):
    """Prediction flips when each ``size`` square is filled with ``fill``.

    Returns ``(rates, maps)``: ``rates`` is the share of images flipped per window
    position (rows x cols grid), ``maps`` an (n, H, W) per-image flip-count map.
    """
    X = torch.as_tensor(images, dtype=torch.float32)
    n, _, H, W = X.shape
    pred = model(X).argmax(1)
    starts_r = list(range(0, H - size + 1, stride))
    starts_c = list(range(0, W - size + 1, stride))
    rates = np.zeros((len(starts_r), len(starts_c)))
    maps = torch.zeros(n, H, W)
    for i, r in enumerate(starts_r):
        for j, c in enumerate(starts_c):
            Xo = X.clone()
            Xo[:, :, r : r + size, c : c + size] = fill
            flipped = (model(Xo).argmax(1) != pred).float()
            rates[i, j] = float(flipped.mean())
            maps[:, r : r + size, c : c + size] += flipped[:, None, None]
    return rates, maps.numpy()
