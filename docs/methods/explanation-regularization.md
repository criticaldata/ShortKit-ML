# Explanation Regularization (M05 RRR)

Explanation Regularization (Right for Right Reasons, RRR) is a **model-level mitigation** (Ross et al. 2017) that penalizes input gradients on shortcut regions during training. The model is discouraged from relying on irrelevant features for predictions.

## Reference

Ross, A. S., Hughes, M. C., & Doshi-Velez, F. (2017). *Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations*. IJCAI 2017. [arXiv:1703.03717](https://arxiv.org/abs/1703.03717)

## What It Does

- **Input:** PyTorch model, images, task labels, shortcut masks (or heatmap file).
- **Training:** For each batch, compute task loss (cross-entropy) plus a penalty on input gradients weighted by the shortcut mask: `L = L_task + λ * Σ(mask ⊙ (∂log p(y|x)/∂x)²)`.
- **Output:** Fine-tuned model checkpoint (.pt).

## Requirements

- PyTorch (already a dependency of shortcut-detect).
- Model must be differentiable end-to-end.
- Shortcut regions must be identified (e.g., via GradCAM, expert annotation).

## Basic Usage

```python
from shortcut_detect import ExplanationRegularization
import torch
import numpy as np

# model, images (N,C,H,W), labels (N,), shortcut_masks (N,H,W) with 1=shortcut
model = ...
images = torch.randn(32, 3, 224, 224)
labels = np.array([0, 1] * 16)
masks = np.random.rand(32, 224, 224)  # 1 where shortcut

rrr = ExplanationRegularization(
    lambda_rrr=1.0,
    lr=1e-4,
    n_epochs=10,
    batch_size=8,
    head="logits",
    random_state=42,
)
rrr.fit(model, images, labels, masks)
# model is updated in-place
```

## Parameters

- **lambda_rrr:** Weight for gradient penalty (default 1.0).
- **lr:** Learning rate (default 1e-4).
- **n_epochs:** Training epochs (default 10).
- **batch_size:** Batch size (default 8).
- **head:** How to extract logits from model output, e.g. "logits" or 0 (default "logits").
- **random_state:** Seed for reproducibility.

## Workflow

1. Run detection (e.g., GradCAM mask overlap) to identify shortcut regions.
2. Create shortcut masks (1 on shortcut, 0 elsewhere).
3. Fit `ExplanationRegularization` with model, images, labels, and masks.
4. Use the fine-tuned model for inference.

## When to Use

- GradCAM or SpRAy indicates the model attends to shortcut regions.
- You have model access and can fine-tune.
- You want to steer the model toward the right features without retraining from scratch.
