"""Procedurally generated image benchmark with a planted shortcut.

No downloads and no licensed data: every image is drawn from a seed.

Task Y: the image holds one faint bar, horizontal (1) or vertical (0), drawn
at a random position, length and width in low contrast on a noisy background.
Shortcut A: a small solid red patch in one corner of the image. The corner is
fixed within a dataset (so a model can learn it) and chosen by the dataset's
seed (so a method cannot hard-code it). A agrees with Y with probability rho;
rho = 0.5 means the patch carries no information about the label.

Images are float32 arrays of shape (n, 3, 32, 32) with values in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = 32
PATCH = 4
MARGIN = 2
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


@dataclass
class ImageDataset:
    images: np.ndarray  # (n, 3, 32, 32)
    task_labels: np.ndarray  # (n,) 1 = horizontal bar, 0 = vertical bar
    attribute: np.ndarray  # (n,) 1 = patch present
    rho: float
    corner: str
    seed: int

    @property
    def patch_box(self) -> tuple[int, int, int, int]:
        return patch_box(self.corner)


def patch_box(corner: str) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1) of the patch, half-open."""
    r0 = MARGIN if corner.startswith("top") else SIZE - MARGIN - PATCH
    c0 = MARGIN if corner.endswith("left") else SIZE - MARGIN - PATCH
    return r0, r0 + PATCH, c0, c0 + PATCH


def generate_shapes(
    n: int,
    rho: float,
    seed: int,
    *,
    corner: str | None = None,
    contrast: float = 0.15,
    noise: float = 0.12,
) -> ImageDataset:
    rng = np.random.default_rng(seed)
    if corner is None:
        corner = CORNERS[int(rng.integers(len(CORNERS)))]
    y = (rng.random(n) < 0.5).astype(np.int64)
    a = np.where(rng.random(n) < rho, y, 1 - y).astype(np.int64)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    img = np.full((n, SIZE, SIZE), 0.5, dtype=np.float32)
    img += rng.normal(0.0, noise, size=img.shape).astype(np.float32)
    # bar centres stay clear of the four corner patch areas
    cy = rng.uniform(11, SIZE - 11, n)
    cx = rng.uniform(11, SIZE - 11, n)
    half_len = rng.uniform(4.0, 6.0, n)
    half_wid = rng.uniform(1.0, 1.8, n)
    for i in range(n):
        h, w = (half_wid[i], half_len[i]) if y[i] == 1 else (half_len[i], half_wid[i])
        img[i][(np.abs(yy - cy[i]) <= h) & (np.abs(xx - cx[i]) <= w)] += contrast
    rgb = np.repeat(img[:, None], 3, axis=1)
    r0, r1, c0, c1 = patch_box(corner)
    on = a == 1
    rgb[on, 0, r0:r1, c0:c1] = 0.95
    rgb[on, 1, r0:r1, c0:c1] = 0.1
    rgb[on, 2, r0:r1, c0:c1] = 0.1
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return ImageDataset(rgb.astype(np.float32), y, a, float(rho), corner, seed)
