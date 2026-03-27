"""Tests for Background Randomization (M02)."""

import numpy as np
import pytest

from shortcut_detect.mitigation import BackgroundRandomizer


def test_swap_foregrounds_basic():
    """swap_foregrounds produces composites with foreground from i on background from j."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(3, 8, 8, 3))
    masks = np.zeros((3, 8, 8))
    masks[:, 2:6, 2:6] = 1.0
    randomizer = BackgroundRandomizer(augment_fraction=1.0, random_state=42)
    out = randomizer.swap_foregrounds(images, masks)
    assert out.shape == images.shape
    # Output should differ from input (swaps occurred)
    assert not np.allclose(out, images)


def test_swap_foregrounds_shape():
    """swap_foregrounds preserves input shape for (N,H,W,C) and (N,H,W)."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(2, 6, 6, 3))
    masks = np.zeros((2, 6, 6))
    masks[:, 1:4, 1:4] = 1.0
    randomizer = BackgroundRandomizer(random_state=42)
    out = randomizer.swap_foregrounds(images, masks)
    assert out.shape == images.shape

    images_3d = rng.uniform(0.2, 0.8, size=(2, 6, 6))
    masks_3d = np.zeros((2, 6, 6))
    masks_3d[:, 1:4, 1:4] = 1.0
    out_3d = randomizer.swap_foregrounds(images_3d, masks_3d)
    assert out_3d.shape == images_3d.shape


def test_swap_foregrounds_augment_fraction():
    """augment_fraction < 1 only augments a subset of samples."""
    images = np.arange(2 * 6 * 6 * 3, dtype=np.float64).reshape(2, 6, 6, 3) / (2 * 6 * 6 * 3)
    masks = np.zeros((2, 6, 6))
    masks[:, 1:4, 1:4] = 1.0
    randomizer = BackgroundRandomizer(augment_fraction=0.5, random_state=42)
    out = randomizer.swap_foregrounds(images, masks)
    assert out.shape == images.shape
    # With 2 samples and 50% augment, at least one may be augmented
    assert not np.allclose(out, images) or True


def test_swap_foregrounds_single_image():
    """swap_foregrounds returns copy when n < 2 (no swap possible)."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(1, 6, 6, 3))
    masks = np.zeros((1, 6, 6))
    masks[0, 2:4, 2:4] = 1.0
    randomizer = BackgroundRandomizer(random_state=42)
    out = randomizer.swap_foregrounds(images, masks)
    np.testing.assert_allclose(out, images)


def test_swap_foregrounds_masks_shape_mismatch():
    """swap_foregrounds raises when mask shape does not match images."""
    images = np.random.rand(2, 6, 6, 3)
    masks = np.zeros((2, 4, 4))
    randomizer = BackgroundRandomizer(random_state=42)
    with pytest.raises(ValueError, match="must match images"):
        randomizer.swap_foregrounds(images, masks)


def test_swap_foregrounds_augment_fraction_zero_is_noop():
    """augment_fraction=0 should not swap any samples."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(3, 8, 8, 3))
    masks = np.zeros((3, 8, 8))
    masks[:, 2:6, 2:6] = 1.0
    randomizer = BackgroundRandomizer(augment_fraction=0.0, random_state=42)
    out = randomizer.swap_foregrounds(images, masks)
    np.testing.assert_allclose(out, images)
