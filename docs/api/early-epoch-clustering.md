# Early Epoch Clustering Detector API

The `EarlyEpochClusteringDetector` detects shortcut bias by clustering early-epoch representations and identifying imbalanced clusters that suggest shortcut reliance.

## Class Reference

::: shortcut_detect.training.EarlyEpochClusteringDetector
    options:
      show_root_heading: true
      show_source: true

## Quick Reference

### Constructor

```python
EarlyEpochClusteringDetector(
    n_clusters: int = 4,
    min_cluster_ratio: float = 0.1,
    entropy_threshold: float = 0.7,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | 4 | Number of clusters to form |
| `min_cluster_ratio` | float | 0.1 | Minimum cluster size ratio for balance check |
| `entropy_threshold` | float | 0.7 | Entropy threshold for shortcut detection |

## Methods

### fit()

```python
def fit(
    early_epoch_reps: np.ndarray,
    labels: np.ndarray = None,
    n_epochs: int = 1,
) -> EarlyEpochClusteringDetector
```

Fit the clustering detector on early-epoch representations.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `early_epoch_reps` | ndarray | Shape (n_samples, n_features), early-epoch representations |
| `labels` | ndarray or None | Optional labels for cluster-label agreement |
| `n_epochs` | int | Number of early epochs the representations come from |

**Returns:** self

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `report_` | EarlyEpochClusteringReport | Report with cluster statistics and risk assessment |

## Usage Examples

### Basic Usage

```python
from shortcut_detect.training import EarlyEpochClusteringDetector

detector = EarlyEpochClusteringDetector(n_clusters=4)
detector.fit(early_epoch_reps, labels=labels, n_epochs=1)
print(detector.report_)
```

### Custom Parameters

```python
detector = EarlyEpochClusteringDetector(
    n_clusters=6,
    min_cluster_ratio=0.05,
    entropy_threshold=0.6,
)
detector.fit(early_epoch_reps)
```

### Via Unified ShortcutDetector

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["early_epoch_clustering"])
detector.fit(embeddings, labels)
print(detector.summary())
```

## See Also

- [Early Epoch Clustering Method Guide](../methods/early-epoch-clustering.md)
- [ShortcutDetector API](shortcut-detector.md)
