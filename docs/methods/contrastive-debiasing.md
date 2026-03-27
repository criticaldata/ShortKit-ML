# Contrastive Debiasing (M07)

Contrastive Debiasing is a **model-level mitigation** (Zhang et al. 2022, Correct-n-Contrast) that uses contrastive learning to separate shortcuts from task-relevant signals. Same-class examples with different spurious (group) attributes are pulled together; different-class examples are pushed apart.

## Reference

Zhang, R., Sharma, A., Li, J., Chen, S., Wang, Y., & Ré, C. (2022). *Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations*. ICML 2022. Contrastive learning with anchor-positive-negative sampling improves worst-group accuracy without requiring group labels at inference.

## What It Does

- **Input:** Embeddings, task labels, and group (protected/spurious) labels.
- **Training:** Anchors are from a (task, group) slice. Positives = same task, different group. Negatives = different task. InfoNCE contrastive loss + optional CE for task preservation. Encoder learns representations invariant to group while preserving task signal.
- **Output:** Debiased embeddings with reduced spurious encoding.

## Requirements

- PyTorch (already a dependency of shortcut-detect).
- NumPy.
- At least 2 task classes and 2 groups (each task must have samples in at least 2 groups).

## Basic Usage

```python
from shortcut_detect import ContrastiveDebiasing, SKLearnProbe
import numpy as np

# Embeddings (n_samples, embed_dim), task labels, and group labels
embeddings = ...  # shape (N, D)
task_labels = ...  # shape (N,), e.g., 0/1 for binary task
group_labels = ...  # shape (N,), e.g., 0/1 for binary spurious attribute

cnc = ContrastiveDebiasing(
    hidden_dim=32,
    temperature=0.05,
    contrastive_weight=0.75,
    use_task_loss=True,
    n_epochs=50,
    batch_size=64,
    random_state=42,
)
cnc.fit(embeddings, task_labels, group_labels)
embeddings_debiased = cnc.transform(embeddings)
# embeddings_debiased has shape (N, hidden_dim)
```

### Verify debiasing

```python
from shortcut_detect import SKLearnProbe

probe = SKLearnProbe(threshold=0.7)
probe.fit(embeddings, group_labels)
acc_before = probe.metric_value_

probe.fit(embeddings_debiased, group_labels)
acc_after = probe.metric_value_

print(f"Probe accuracy before: {acc_before:.2%}")
print(f"Probe accuracy after: {acc_after:.2%}")  # Should drop significantly
```

## Parameters

- **hidden_dim:** Hidden dimension of the encoder (default: min(64, embed_dim)).
- **temperature:** Temperature for InfoNCE contrastive loss; lower = sharper (default 0.05).
- **contrastive_weight:** Weight for contrastive loss vs CE; 1.0 = pure contrastive (default 0.75).
- **use_task_loss:** If True, jointly minimize CE for task prediction (default True).
- **n_epochs:** Training epochs (default 50).
- **batch_size:** Batch size (default 64).
- **lr:** Learning rate (default 1e-3).
- **dropout:** Dropout rate in encoder (default 0.1).
- **random_state:** Seed for reproducibility.

## Workflow

1. Run detection (e.g., Probe) to confirm spurious encoding.
2. Fit `ContrastiveDebiasing` on embeddings, task labels, and group labels.
3. Use `transform()` to obtain debiased embeddings for downstream tasks.
4. Re-run Probe on group labels to verify reduced spurious predictability.

## When to Use

- Probe or geometric analysis indicates strong spurious/group encoding.
- You have both task and group labels.
- You want model-level mitigation that aligns same-class/different-group representations.
- Correct-n-Contrast is suited for worst-group robustness when spurious correlations exist.
