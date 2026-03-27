# Last Layer Retraining (M06 DFR)

Last Layer Retraining (DFR, Deep Feature Reweighting) is a **model-level mitigation** (Kirichenko et al. 2023) that retrains only the last linear layer on a group-balanced subset of embeddings. The embeddings stay frozen; only the classifier is retrained. This simple approach can match or outperform more complex debiasing methods.

## Reference

Kirichenko, P., Izmailov, P., & Wilson, A. G. (2023). *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*. ICLR 2023. [arXiv:2204.02937](https://arxiv.org/abs/2204.02937)

## What It Does

- **Input:** Embeddings, task labels (target to predict), and group/protected labels.
- **Training:** Build a group-balanced subset (equal samples per group), then fit a LogisticRegression classifier on this subset. StandardScaler preprocesses embeddings.
- **Output:** Trained classifier that predicts task labels. Output CSV includes original data plus `dfr_prediction` column.

## Requirements

- scikit-learn (already a dependency of shortcut-detect).
- NumPy.

## Basic Usage

```python
from shortcut_detect import LastLayerRetraining
import numpy as np

# Embeddings (n_samples, embed_dim), task labels, and group labels
embeddings = ...  # shape (N, D)
task_labels = ...  # shape (N,), e.g., class 0/1
group_labels = ...  # shape (N,), e.g., demographic group 0/1/2

dfr = LastLayerRetraining(
    C=1.0,
    penalty="l1",
    class_weight="balanced",
    random_state=42,
)
dfr.fit(embeddings, task_labels, group_labels)
predictions = dfr.predict(embeddings)
```

### Convenience: fit_predict

```python
predictions = dfr.fit_predict(embeddings, task_labels, group_labels)
```

## Parameters

- **C:** Inverse regularization strength (default 1.0). Smaller = stronger regularization.
- **penalty:** "l1" or "l2" (default "l1").
- **solver:** Solver for LogisticRegression (default "liblinear").
- **class_weight:** "balanced" or None for imbalanced task labels (default "balanced").
- **random_state:** Seed for reproducibility.

## Workflow

1. Run detection and export embeddings with task_label and group_label.
2. Fit `LastLayerRetraining` on embeddings, task labels, and group labels.
3. Use `predict()` to obtain task predictions for new or existing embeddings.
4. Compare worst-group accuracy before vs. after DFR.

## When to Use

- Spurious correlation between group and task (e.g., Waterbirds, CelebA).
- You have embeddings and group labels but cannot retrain the full model.
- You want a simple, fast mitigation (sklearn only) vs. adversarial training (M04).
