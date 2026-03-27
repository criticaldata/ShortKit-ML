# Causal Effect Regularization

Causal Effect Regularization (Kumar et al., 2023) estimates the causal effect of each attribute on the task label. Attributes with near-zero estimated effect are flagged as spurious shortcuts.

## What It Detects

- Per-attribute causal effect on the task label.
- Attributes with low causal effect (spurious) that the classifier should not rely on.

## Required Inputs

- `embeddings`: `np.ndarray` `(n, d)` representation space
- `labels`: `np.ndarray` `(n,)` task labels
- `attributes`: `dict[str, np.ndarray]` - attribute name -> `(n,)` values per sample (binary or categorical)

Optional:

- `counterfactual_pairs`: For interventional data (Phase 2). Not used in current Direct estimator.

## Unified API Example

```python
from shortcut_detect import ShortcutDetector

loader = {
    "embeddings": emb,
    "labels": labels,
    "attributes": {
        "race": race_labels,
        "color": color_labels,
    },
}

detector = ShortcutDetector(
    methods=["causal_effect"],
    causal_effect_spurious_threshold=0.1,
)
detector.fit_from_loaders({"causal_effect": loader})

result = detector.get_results()["causal_effect"]
print(result["metrics"])
print(result["report"]["per_attribute"])
```

## Interpretation

- Attributes with |TE_a| < threshold are flagged as spurious (shortcuts).
- Higher estimated causal effect indicates the attribute may be task-relevant.
- Risk levels:
  - `high`: multiple spurious attributes
  - `moderate`: one spurious attribute
  - `low`: no spurious attributes

## Reference

Kumar, Abhinav, Amit Deshpande, and Amit Sharma. "Causal Effect Regularization: Automated Detection and Removal of Spurious Attributes." arXiv:2306.11072 (2023).
