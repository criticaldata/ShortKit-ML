# SpRAy (Heatmap Clustering)

SpRAy (Spectral Relevance Analysis) clusters explanation heatmaps to reveal
systematic attention patterns that can indicate Clever Hans behavior.

This implementation follows Lapuschkin et al. (2019) and is designed to work
with any heatmap generator (e.g., GradCAM). You can provide precomputed
heatmaps directly or let SpRAy generate them via a pluggable heatmap generator.

## When to Use

Use SpRAy when:
- You can compute explanation heatmaps for a set of samples.
- You want to discover *systematic* attention artifacts.
- You need an interpretable, cluster-based view of model focus.

Do not use SpRAy when:
- You have very few samples (spectral clustering becomes unstable).
- Heatmaps are extremely noisy or poorly aligned across samples.

## Example (Precomputed Heatmaps)

```python
import numpy as np
from shortcut_detect import SpRAyDetector

# heatmaps: (N, H, W)
heatmaps = np.load("heatmaps.npy")
labels = np.load("labels.npy")

detector = SpRAyDetector(
    affinity="cosine",
    cluster_selection="auto",
    downsample_size=32,
)
detector.fit(heatmaps=heatmaps, labels=labels)
report = detector.get_report()

print(report["report"]["clever_hans"])
```

## Example (Generate Heatmaps via GradCAM)

```python
import torch
from shortcut_detect import SpRAyDetector

model = torch.load("model.pt", map_location="cpu", weights_only=False)
inputs = ...  # tensor of shape (N, C, H, W)

detector = SpRAyDetector(
    affinity="rbf",
    cluster_selection="eigengap",
    max_clusters=8,
)
detector.fit(
    inputs=inputs,
    model=model,
    target_layer="backbone.layer4",
    head="logits",
)
print(detector.summary())
```

## Key Outputs

- **clusters**: size, purity, focus (localization) stats per cluster
- **clever_hans**: heuristic flags + risk level
- **representative_heatmaps**: mean heatmap per cluster for inspection

## Tips

- Start with `affinity="cosine"` and `cluster_selection="auto"`.
- Use `downsample_size=32` or `64` to speed up clustering.
- If you see small, high-purity clusters with very localized attention,
  investigate corresponding inputs for artifacts.
