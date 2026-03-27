"""Tests for Shortcut Feature Masking (M01)."""

import numpy as np
import pytest

from shortcut_detect.mitigation import ShortcutMasker


def test_shortcut_masker_init():
    """ShortcutMasker accepts valid strategy and rejects invalid."""
    ShortcutMasker(strategy="zero")
    ShortcutMasker(strategy="randomize")
    ShortcutMasker(strategy="inpaint")
    with pytest.raises(ValueError, match="strategy must be"):
        ShortcutMasker(strategy="invalid")


def test_mask_images_zero():
    """mask_images with strategy zero sets shortcut region to 0."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(2, 8, 8, 3))
    masks = np.zeros((2, 8, 8))
    masks[:, 2:6, 2:6] = 1.0
    masker = ShortcutMasker(strategy="zero", random_state=42)
    out = masker.mask_images(images, shortcut_masks=masks)
    assert out.shape == images.shape
    np.testing.assert_allclose(out[:, 2:6, 2:6, :], 0.0)
    np.testing.assert_allclose(out[:, :2, :, :], images[:, :2, :, :])
    np.testing.assert_allclose(out[:, 6:, :, :], images[:, 6:, :, :])


def test_mask_images_randomize():
    """mask_images with strategy randomize changes shortcut region."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(2, 8, 8, 3))
    masks = np.zeros((2, 8, 8))
    masks[:, 2:6, 2:6] = 1.0
    masker = ShortcutMasker(strategy="randomize", random_state=42)
    out = masker.mask_images(images, shortcut_masks=masks)
    assert out.shape == images.shape
    # Shortcut region should differ from original
    assert not np.allclose(out[:, 2:6, 2:6, :], images[:, 2:6, 2:6, :])
    # Non-mask region unchanged
    np.testing.assert_allclose(out[:, :2, :, :], images[:, :2, :, :])


def test_mask_images_heatmaps():
    """mask_images with heatmaps binarizes by threshold."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(1, 6, 6, 3))
    heatmaps = np.zeros((1, 6, 6))
    heatmaps[0, 1:4, 1:4] = 0.6
    masker = ShortcutMasker(strategy="zero", heatmap_threshold=0.5, random_state=42)
    out = masker.mask_images(images, heatmaps=heatmaps)
    assert out.shape == images.shape
    np.testing.assert_allclose(out[0, 1:4, 1:4, :], 0.0)


def test_mask_images_inpaint():
    """mask_images with strategy inpaint fills shortcut region."""
    images = np.ones((1, 6, 6, 3)) * 0.5
    masks = np.zeros((1, 6, 6))
    masks[0, 2:4, 2:4] = 1.0
    masker = ShortcutMasker(strategy="inpaint", random_state=42)
    out = masker.mask_images(images, shortcut_masks=masks)
    assert out.shape == images.shape
    # Inpaint fills with mean of unmasked (0.5), so region should be 0.5
    np.testing.assert_allclose(out[0, 2:4, 2:4, :], 0.5)


def test_mask_images_3d_grayscale():
    """mask_images preserves (N, H, W) input shape."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(1, 6, 6))
    masks = np.zeros((1, 6, 6))
    masks[0, 1:3, 1:3] = 1.0
    masker = ShortcutMasker(strategy="zero", random_state=42)
    out = masker.mask_images(images, shortcut_masks=masks)
    assert out.shape == (1, 6, 6)
    np.testing.assert_allclose(out[0, 1:3, 1:3], 0.0)


def test_mask_embeddings_zero():
    """mask_embeddings with strategy zero zeros flagged dimensions."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(10, 5))
    masker = ShortcutMasker(strategy="zero", random_state=42)
    out = masker.mask_embeddings(emb, [0, 2])
    assert out.shape == emb.shape
    np.testing.assert_allclose(out[:, 0], 0.0)
    np.testing.assert_allclose(out[:, 2], 0.0)
    np.testing.assert_allclose(out[:, 1], emb[:, 1])
    np.testing.assert_allclose(out[:, 3], emb[:, 3])
    np.testing.assert_allclose(out[:, 4], emb[:, 4])


def test_mask_embeddings_randomize():
    """mask_embeddings with strategy randomize shuffles flagged dimensions."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(20, 4))
    masker = ShortcutMasker(strategy="randomize", random_state=42)
    out = masker.mask_embeddings(emb, [1])
    assert out.shape == emb.shape
    # Column 1 should be a permutation of original
    np.testing.assert_allclose(np.sort(out[:, 1]), np.sort(emb[:, 1]))
    # Other columns unchanged
    np.testing.assert_allclose(out[:, 0], emb[:, 0])
    np.testing.assert_allclose(out[:, 2], emb[:, 2])
    np.testing.assert_allclose(out[:, 3], emb[:, 3])


def test_mask_embeddings_augment_fraction():
    """augment_fraction < 1 only augments a subset of samples."""
    emb = np.arange(20 * 4, dtype=np.float64).reshape(20, 4)
    masker = ShortcutMasker(strategy="zero", augment_fraction=0.5, random_state=42)
    out = masker.mask_embeddings(emb, [0])
    assert out.shape == emb.shape
    # Some rows have dim 0 zeroed, some unchanged
    zeroed = np.where(out[:, 0] == 0)[0]
    unchanged = np.where(out[:, 0] != 0)[0]
    assert len(zeroed) >= 1
    assert len(unchanged) >= 1
    np.testing.assert_allclose(out[unchanged, :], emb[unchanged, :])


def test_mask_embeddings_empty_flagged():
    """mask_embeddings with no valid dim indices returns copy."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(5, 3))
    masker = ShortcutMasker(strategy="zero", random_state=42)
    out = masker.mask_embeddings(emb, [10, -1])
    np.testing.assert_allclose(out, emb)


def test_mask_images_requires_masks_or_heatmaps():
    """mask_images raises if neither shortcut_masks nor heatmaps provided."""
    images = np.random.rand(2, 4, 4, 3)
    masker = ShortcutMasker(strategy="zero")
    with pytest.raises(ValueError, match="Provide either"):
        masker.mask_images(images)


def test_mask_images_augment_fraction_zero_is_noop():
    """augment_fraction=0 should not modify image samples."""
    rng = np.random.default_rng(42)
    images = rng.uniform(0.2, 0.8, size=(3, 8, 8, 3))
    masks = np.zeros((3, 8, 8))
    masks[:, 2:6, 2:6] = 1.0
    masker = ShortcutMasker(strategy="zero", augment_fraction=0.0, random_state=42)
    out = masker.mask_images(images, shortcut_masks=masks)
    np.testing.assert_allclose(out, images)


def test_mask_embeddings_augment_fraction_zero_is_noop():
    """augment_fraction=0 should not modify embedding samples."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(size=(10, 5))
    masker = ShortcutMasker(strategy="zero", augment_fraction=0.0, random_state=42)
    out = masker.mask_embeddings(emb, [0, 2])
    np.testing.assert_allclose(out, emb)
