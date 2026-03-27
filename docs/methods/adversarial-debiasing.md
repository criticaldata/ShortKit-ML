# Adversarial Debiasing (M04)

Adversarial Debiasing is a **model-level mitigation** (Zhang et al. 2018) that uses adversarial training to remove demographic encoding from embeddings. The encoder learns representations that predict the task while being uninformative for the protected attribute, via a Gradient Reversal Layer (GRL).

## Reference

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). *Mitigating Unwanted Biases with Adversarial Learning*. AIES 2018. The predictor and adversary are trained jointly so the predictor remains accurate while exhibiting less stereotyping.

## What It Does

- **Input:** Embeddings, protected (demographic) labels, and optionally task labels.
- **Training:** Encoder maps embeddings to a hidden representation. An adversary tries to predict the protected attribute from this representation via a GRL. The encoder learns to minimize task loss while maximizing adversary loss (making protected attribute unpredictable).
- **Output:** Debiased embeddings with reduced demographic encoding.

## Requirements

- PyTorch (already a dependency of shortcut-detect).
- NumPy.

## Basic Usage

### Without task labels (demographic removal only)

```python
from shortcut_detect import AdversarialDebiasing, SKLearnProbe
import numpy as np

# Embeddings (n_samples, embed_dim) and protected labels (e.g., gender)
embeddings = ...  # shape (N, D)
protected_labels = ...  # shape (N,), e.g., 0/1 for binary

debiaser = AdversarialDebiasing(
    hidden_dim=64,
    adversary_weight=0.5,
    n_epochs=50,
    batch_size=64,
    random_state=42,
)
debiaser.fit(embeddings, protected_labels)
embeddings_debiased = debiaser.transform(embeddings)
# embeddings_debiased has shape (N, hidden_dim)
```

### With task labels (preserve utility)

```python
debiaser = AdversarialDebiasing(
    hidden_dim=64,
    adversary_weight=0.5,
    n_epochs=50,
)
debiaser.fit(embeddings, protected_labels, task_labels=task_labels)
embeddings_debiased = debiaser.transform(embeddings)
```

### Verify debiasing

```python
from shortcut_detect import SKLearnProbe

probe = SKLearnProbe(threshold=0.7)
probe.fit(embeddings, protected_labels)
acc_before = probe.metric_value_

probe.fit(embeddings_debiased, protected_labels)
acc_after = probe.metric_value_

print(f"Probe accuracy before: {acc_before:.2%}")
print(f"Probe accuracy after: {acc_after:.2%}")  # Should drop significantly
```

## Parameters

- **hidden_dim:** Hidden dimension of the encoder (default: min(64, embed_dim)).
- **adversary_weight:** Weight for adversarial loss; higher removes more demographic info (default 0.5).
- **n_epochs:** Training epochs (default 50).
- **batch_size:** Batch size (default 64).
- **lr:** Learning rate (default 1e-3).
- **dropout:** Dropout rate in encoder (default 0.1).
- **random_state:** Seed for reproducibility.

## Workflow

1. Run detection (e.g., Probe) to confirm demographic encoding.
2. Fit `AdversarialDebiasing` on embeddings and protected labels.
3. Use `transform()` to obtain debiased embeddings for downstream tasks.
4. Re-run Probe to verify reduced demographic predictability.

## When to Use

- Probe or geometric analysis indicates high demographic encoding.
- You need model-level mitigation (vs. data augmentation like M01/M02).
- You want to retain task utility while removing demographic information.
