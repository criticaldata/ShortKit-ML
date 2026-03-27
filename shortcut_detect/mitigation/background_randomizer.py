"""
Background Randomization (M02) - Kwon et al. 2023.

Swap foregrounds with random backgrounds to produce augmented training data
that reduces reliance on background shortcuts.
"""

from __future__ import annotations

import numpy as np


class BackgroundRandomizer:
    """
    Swap foregrounds with random backgrounds across samples (Kwon et al. 2023).

    For each selected sample i, pick a random j != i and compose:
    augmented_i = background_j + foreground_i (foreground of i pasted onto background of j).

    Parameters
    ----------
    augment_fraction : float
        Fraction of samples to augment (0–1). 1.0 = all samples.
    random_state : Optional[int]
        Seed for reproducible randomization.
    """

    def __init__(
        self,
        augment_fraction: float = 1.0,
        random_state: int | None = None,
    ):
        self.augment_fraction = float(augment_fraction)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def swap_foregrounds(
        self,
        images: np.ndarray,
        foreground_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Produce augmented images by swapping foregrounds onto random backgrounds.

        Parameters
        ----------
        images : np.ndarray
            Images of shape (N, H, W) or (N, H, W, C), values in [0, 1].
        foreground_masks : np.ndarray
            Binary masks (N, H, W), 1 = foreground, 0 = background.

        Returns
        -------
        np.ndarray
            Augmented images, same shape as images.
        """
        images = np.asarray(images, dtype=np.float64)
        foreground_masks = np.asarray(foreground_masks, dtype=np.float64)
        need_squeeze = images.ndim == 3
        if need_squeeze:
            images = images[:, :, :, np.newaxis]
        if foreground_masks.ndim == 2:
            foreground_masks = foreground_masks[np.newaxis, ...]
        n, h, w, c = images.shape
        if (
            foreground_masks.shape[0] != n
            or foreground_masks.shape[1] != h
            or foreground_masks.shape[2] != w
        ):
            raise ValueError(
                f"foreground_masks shape {foreground_masks.shape} must match images (N={n}, H={h}, W={w})"
            )
        masks = (foreground_masks > 0.5).astype(np.float64)
        masks_exp = masks[:, :, :, np.newaxis] if masks.ndim == 3 else masks

        if n < 2:
            if need_squeeze:
                return images.squeeze(axis=-1)
            return images.copy()

        if self.augment_fraction < 1.0:
            n_aug = int(n * self.augment_fraction)
            if n_aug <= 0:
                if need_squeeze:
                    return images.squeeze(axis=-1)
                return images.copy()
            indices = self._rng.choice(n, size=n_aug, replace=False)
        else:
            indices = np.arange(n)

        out = images.copy()
        for i in indices:
            j = self._rng.integers(0, n)
            while j == i and n > 1:
                j = self._rng.integers(0, n)
            background_j = images[j] * (1.0 - masks_exp[j])
            foreground_i = images[i] * masks_exp[i]
            out[i] = background_j + foreground_i

        if need_squeeze:
            out = out.squeeze(axis=-1)
        return np.clip(out, 0.0, 1.0).astype(np.float64)
