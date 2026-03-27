# Background Randomization (M02)

Background Randomization is a **data mitigation** approach (Kwon et al. 2023) that swaps foregrounds with random backgrounds across samples. This breaks spurious object–background correlations so models cannot rely on background shortcuts.

## Reference

Kwon et al. (ICLR 2023): "Disentangled Feature Swapping Augmentation for Weakly Supervised Semantic Segmentation." A pixel-level approximation composes new images by pasting the foreground (masked region) of one image onto the background of another.

## What It Does

Given images and foreground masks, for each sample i the method picks a random sample j (j != i) and produces: augmented_i = background_j + foreground_i. The foreground of image i is pasted onto the background of image j.

## Requirements

NumPy and PIL only. No extra dependencies.

## Basic Usage

```python
from shortcut_detect import BackgroundRandomizer
import numpy as np

randomizer = BackgroundRandomizer(augment_fraction=1.0, random_state=42)
# images: (N, H, W) or (N, H, W, C), values in [0, 1]
# foreground_masks: (N, H, W), 1 = foreground, 0 = background
augmented = randomizer.swap_foregrounds(images, foreground_masks)
```

## Dashboard

In the **Advanced Analysis** tab, use the **Background Randomization (M02)** accordion:

- Upload images and either foreground mask images or a heatmap file (.npz/.npy).
- Set heatmap threshold and augment fraction.
- Download a zip of augmented images.

Requires at least 2 images to perform swaps.

## Workflow

1. Obtain foreground masks (from GradCAM, segmentation model, or manual annotation).
2. Run Background Randomization to produce augmented images.
3. Use the augmented data for retraining.

## Parameters

- **augment_fraction:** Fraction of samples to augment, 0–1 (default 1.0).
- **random_state:** Seed for reproducible randomization.
