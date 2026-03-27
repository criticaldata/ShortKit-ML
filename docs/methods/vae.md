# VAE (Variational Autoencoder) Shortcut Detection

VAE-based shortcut detection (Müller et al., Fraunhofer-AISEC) uses Beta-VAE disentanglement
to identify latent dimensions with high predictiveness for the target label.

## Requirements

- `torch` and `torchvision` (included in core install; see [Installation](../getting-started/installation.md))

Reference: Müller et al., "Shortcut Detection with Variational Autoencoders", ICML 2023
Workshop on Spurious Correlations, Invariance and Stability.
[GitHub](https://github.com/Fraunhofer-AISEC/shortcut-detection-vae)

## What It Detects

- Latent dimensions that are highly predictive of the target label (classifier weights).
- High predictiveness indicates the dimension may encode a shortcut (spurious correlation).

## Required Inputs

- `images`: `np.ndarray` or `torch.Tensor` `(N, C, H, W)` or `(N, H, W, C)`
- `labels`: `np.ndarray` `(N,)` class labels
- `img_size`: `int` — image height/width (assume square)

**Or** provide DataLoaders instead:

- `train_dl`, `val_dl`: PyTorch DataLoaders
- `test_dl`: optional, for latent extraction (defaults to `val_dl`)
- `img_size`, `channels`, `num_classes`

## Optional Inputs

- `channels`: default 3 (RGB)
- `num_classes`: default 2
- `vae_checkpoint`: path to pre-trained VAE (skip training)
- `device`: `"cuda:0"` or `"cpu"`

## Unified API Example

```python
from shortcut_detect import ShortcutDetector
import torch

# Using numpy/tensor arrays
images = torch.randn(200, 3, 64, 64)  # or np.ndarray
labels = (torch.rand(200) > 0.5).long().numpy()

bundle = {
    "images": images,
    "labels": labels,
    "img_size": 64,
    "channels": 3,
    "num_classes": 2,
}

detector = ShortcutDetector(
    methods=["vae"],
    vae_latent_dim=10,
    vae_kld_weight=3.0,
    vae_epochs=50,
)
detector.fit_from_loaders({"vae": bundle})

result = detector.get_results()["vae"]
print(result["metrics"])
print(result["report"]["per_dimension"])
```

## Interpretation

- **Predictiveness**: Sum of absolute classifier weights per latent dimension. High values
  indicate the dimension is used for classification (candidate shortcut).
- **MPWD** (max pairwise Wasserstein distance): Class separability per dimension.
- **Flagged**: Dimensions where normalized predictiveness exceeds the threshold (default 0.5).
- Risk levels:
  - `high`: many dimensions flagged (≥ half of latent dims)
  - `moderate`: at least one dimension flagged
  - `low`: no dimensions flagged

## Reference

Müller, Nicolas M., Simon Roschmann, Shahbaz Khan, Philip Sperl, and Konstantin Böttinger.
"Shortcut Detection with Variational Autoencoders." ICML 2023 Workshop on Spurious
Correlations, Invariance and Stability.
