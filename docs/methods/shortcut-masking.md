# Shortcut Feature Masking (M01)

Shortcut Feature Masking is a **data mitigation** approach (Teso & Kersting 2019) that produces augmented training data by masking or inpainting detected shortcut regions. Using this data for retraining discourages models from relying on shortcuts.

## Reference

Teso, S., & Kersting, K. (2019). *Explanatory Interactive Machine Learning*. AIES 2019. The counterexample strategy randomizes or masks wrongly identified components so the learner does not depend on them.

## What It Does

- **Images:** Given images and shortcut regions (from GradCAM heatmaps or user-provided masks), produces new images with those regions zeroed, randomized, or inpainted.
- **Embeddings:** Given embeddings and flagged dimension indices (e.g. from HBAC or statistical detection), produces new embeddings with those dimensions zeroed or randomized.

## Strategies

| Strategy    | Images                         | Embeddings              |
|------------|---------------------------------|--------------------------|
| **zero**   | Set shortcut pixels to 0        | Set flagged dimensions to 0 |
| **randomize** | Replace shortcut pixels with random values | Shuffle values across samples for flagged dims |
| **inpaint**   | Fill shortcut regions with mean (simple inpainting) | N/A (treated as zero) |

## Requirements

- **Image mode:** NumPy, PIL. No extra dependencies.
- **Embedding mode:** NumPy only.

## Basic Usage

### Image mode

```python
from shortcut_detect import ShortcutMasker
import numpy as np

masker = ShortcutMasker(strategy="randomize", heatmap_threshold=0.5, augment_fraction=1.0)
# images: (N, H, W) or (N, H, W, C), values in [0, 1]
# shortcut_masks: (N, H, W) binary, or pass heatmaps instead
augmented = masker.mask_images(images, shortcut_masks=masks)
# Or from GradCAM heatmaps:
augmented = masker.mask_images(images, heatmaps=heatmaps)
```

### Embedding mode

```python
masker = ShortcutMasker(strategy="zero")
# embeddings: (N, D), flagged_dim_indices: list of dimension indices
augmented = masker.mask_embeddings(embeddings, flagged_dim_indices=[0, 3, 7])
```

## Dashboard

In the **Advanced Analysis** tab, use the **Shortcut Feature Masking (M01)** accordion:

- **Image mode:** Upload images and either mask images or a heatmap file (.npz/.npy). Choose strategy (zero / randomize / inpaint) and augment fraction. Download a zip of augmented images.
- **Embedding mode:** Upload an embeddings CSV and enter dimension indices to mask (e.g. `0,3,7`). Download the augmented CSV.

## Workflow

1. Run detection (e.g. GradCAM, HBAC, Statistical) to identify shortcut regions or dimensions.
2. Run Shortcut Masking with those regions/dimensions to produce augmented data.
3. Use the augmented data (optionally mixed with original) for retraining.

## Parameters

- **strategy:** `"zero"` | `"randomize"` | `"inpaint"` (images); `"zero"` | `"randomize"` (embeddings).
- **heatmap_threshold:** Binarization threshold for heatmaps (default 0.5).
- **augment_fraction:** Fraction of samples to augment, 0–1 (default 1.0).
- **random_state:** Seed for reproducible randomization.
