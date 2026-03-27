# Early-Epoch Clustering (SPARE)

Early-epoch clustering detects shortcut bias by exploiting the tendency of SGD-trained models
to learn simple (often spurious) features early in training. By clustering representations or
logits from the first few epochs, we can identify imbalanced clusters that suggest shortcut
reliance.

## When to Use

- You can capture early-epoch representations (logits or features).
- You lack protected-group labels but want a proxy signal.
- You want a training-time shortcut signal before full convergence.

## Usage

```python
from shortcut_detect.training import EarlyEpochClusteringDetector

# early_epoch_reps: array of shape (n_samples, n_features)
# labels optional; used only for cluster-label agreement

detector = EarlyEpochClusteringDetector(
    n_clusters=4,
    min_cluster_ratio=0.1,
    entropy_threshold=0.7,
)

detector.fit(early_epoch_reps, labels=labels, n_epochs=1)
print(detector.report_)
```

## Interpretation

- **Low entropy / low minority ratio** → clusters are imbalanced, suggesting shortcut bias.
- **High entropy / balanced clusters** → no strong shortcut signal detected.

## Reference

Yang et al. 2023, SPARE (SePArate early and REsample).
