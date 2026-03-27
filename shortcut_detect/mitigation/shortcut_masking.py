"""
Shortcut Feature Masking (M01) - Teso & Kersting 2019.

Mask or inpaint detected shortcut regions to produce augmented training data
that discourages models from relying on shortcuts during retraining.
"""

from __future__ import annotations

import numpy as np


class ShortcutMasker:
    """
    Mask or randomize detected shortcut regions (images) or dimensions (embeddings).

    Implements the data augmentation mitigation from Teso & Kersting (2019):
    counterexamples are created by randomizing or zeroing shortcut components.

    Parameters
    ----------
    strategy : str
        For images: "zero", "randomize", or "inpaint".
        For embeddings: "zero" or "randomize".
    heatmap_threshold : float
        Binarization threshold for heatmaps when converting to shortcut masks (0–1).
    augment_fraction : float
        Fraction of samples to augment (0–1). 1.0 = all samples.
    random_state : Optional[int]
        Seed for reproducible randomization.
    """

    def __init__(
        self,
        strategy: str = "randomize",
        heatmap_threshold: float = 0.5,
        augment_fraction: float = 1.0,
        random_state: int | None = None,
    ):
        if strategy not in ("zero", "randomize", "inpaint"):
            raise ValueError("strategy must be 'zero', 'randomize', or 'inpaint'")
        self.strategy = strategy
        self.heatmap_threshold = float(heatmap_threshold)
        self.augment_fraction = float(augment_fraction)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def mask_images(
        self,
        images: np.ndarray,
        shortcut_masks: np.ndarray | None = None,
        heatmaps: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Produce augmented images by masking shortcut regions.

        Parameters
        ----------
        images : np.ndarray
            Images of shape (N, H, W) or (N, H, W, C), values in [0, 1] or [0, 255].
        shortcut_masks : np.ndarray, optional
            Binary masks (N, H, W), 1 = shortcut region. If None, heatmaps are used.
        heatmaps : np.ndarray, optional
            Heatmaps (N, H, W) in [0, 1]. Used if shortcut_masks is None; binarized
            with heatmap_threshold.

        Returns
        -------
        np.ndarray
            Augmented images, same shape and dtype as images.
        """
        images_arr = np.asarray(images)
        orig_dtype = images_arr.dtype
        images = np.asarray(images_arr, dtype=np.float64)
        need_squeeze = images.ndim == 3
        if need_squeeze:
            images = images[:, :, :, np.newaxis]
        n, h, w, c = images.shape
        max_val = 1.0 if images.max() <= 1.0 else 255.0

        if shortcut_masks is None and heatmaps is None:
            raise ValueError("Provide either shortcut_masks or heatmaps")
        if shortcut_masks is None:
            heatmaps = np.asarray(heatmaps, dtype=np.float64)
            if heatmaps.ndim == 2:
                heatmaps = heatmaps[np.newaxis, ...]
            if heatmaps.shape[0] != n or heatmaps.shape[1] != h or heatmaps.shape[2] != w:
                raise ValueError(
                    f"heatmaps shape {heatmaps.shape} must match images (N={n}, H={h}, W={w})"
                )
            shortcut_masks = (heatmaps >= self.heatmap_threshold).astype(np.float64)
        else:
            shortcut_masks = np.asarray(shortcut_masks, dtype=np.float64)
            if shortcut_masks.ndim == 2:
                shortcut_masks = shortcut_masks[np.newaxis, ...]
            shortcut_masks = (shortcut_masks > 0.5).astype(np.float64)
            if (
                shortcut_masks.shape[0] != n
                or shortcut_masks.shape[1] != h
                or shortcut_masks.shape[2] != w
            ):
                raise ValueError(
                    f"shortcut_masks shape {shortcut_masks.shape} must match images (N={n}, H={h}, W={w})"
                )

        # Optionally apply only to a fraction of samples
        if self.augment_fraction < 1.0:
            n_aug = int(n * self.augment_fraction)
            if n_aug <= 0:
                return images_arr.copy()
            indices = self._rng.choice(n, size=n_aug, replace=False)
        else:
            indices = np.arange(n)

        out = images.copy()
        for i in indices:
            mask = shortcut_masks[i]  # (H, W)
            if mask.sum() == 0:
                continue
            # (H, W, C)
            region = out[i]
            if self.strategy == "zero":
                out[i] = region * (1 - mask[:, :, np.newaxis])
            elif self.strategy == "randomize":
                rand = self._rng.random((h, w, c))
                out[i] = region * (1 - mask[:, :, np.newaxis]) + rand * mask[:, :, np.newaxis]
            elif self.strategy == "inpaint":
                out[i] = self._inpaint_region(region, mask)
            else:
                raise ValueError("strategy must be 'zero', 'randomize', or 'inpaint'")

        if need_squeeze:
            out = out.squeeze(axis=-1)
        out = np.clip(out, 0.0, max_val)
        try:
            out = out.astype(orig_dtype)
        except (ValueError, TypeError):
            out = out.astype(np.float32)
        return out

    def _inpaint_region(self, region: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Simple inpainting: fill masked pixels with mean of unmasked neighborhood (blur-like)."""
        h, w, c = region.shape
        out = region.copy()
        mask_flat = mask > 0.5
        if not mask_flat.any():
            return out
        # Mean of entire image (per channel) as fill
        for ch in range(c):
            channel = region[:, :, ch]
            fill_val = np.mean(channel[~mask_flat]) if (~mask_flat).any() else 0.0
            out[:, :, ch][mask_flat] = fill_val
        return out

    def mask_embeddings(
        self,
        embeddings: np.ndarray,
        flagged_dim_indices: list[int] | np.ndarray,
    ) -> np.ndarray:
        """
        Produce augmented embeddings by masking flagged shortcut dimensions.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, D).
        flagged_dim_indices : list or array of int
            Dimension indices to mask (0-based).

        Returns
        -------
        np.ndarray
            Augmented embeddings, same shape as embeddings.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        n, d = embeddings.shape
        dims = np.asarray(flagged_dim_indices, dtype=np.intp)
        dims = dims[(dims >= 0) & (dims < d)]
        if len(dims) == 0:
            return embeddings.copy()

        if self.augment_fraction < 1.0:
            n_aug = int(n * self.augment_fraction)
            if n_aug <= 0:
                return embeddings.copy()
            indices = self._rng.choice(n, size=n_aug, replace=False)
        else:
            indices = np.arange(n)

        out = embeddings.copy()
        if self.strategy == "zero":
            out[np.ix_(indices, dims)] = 0.0
        elif self.strategy == "randomize":
            # Shuffle values across samples for flagged dims (Teso & Kersting style)
            for dim in dims:
                col = out[indices, dim].copy()
                self._rng.shuffle(col)
                out[indices, dim] = col
        else:
            # "inpaint" not defined for embeddings; treat as zero
            out[np.ix_(indices, dims)] = 0.0
        return out.astype(embeddings.dtype)
