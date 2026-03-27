# HBACDetector API

The `HBACDetector` class implements Hierarchical Bias-Aware Clustering for shortcut detection.

## Class Reference

::: shortcut_detect.clustering.HBACDetector
    options:
      show_root_heading: true
      show_source: true

## Quick Reference

### Constructor

```python
HBACDetector(
    max_iterations: int = 3,
    min_cluster_size: float = 0.05,
    linkage: str = 'ward',
    distance_metric: str = 'euclidean',
    random_state: int = None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 3 | Maximum depth of hierarchical analysis |
| `min_cluster_size` | float | 0.05 | Minimum cluster size as fraction of total |
| `linkage` | str | 'ward' | Linkage method for clustering |
| `distance_metric` | str | 'euclidean' | Distance metric |
| `random_state` | int | None | Random seed |

### Linkage Options

| Value | Description |
|-------|-------------|
| `'ward'` | Minimize within-cluster variance (default) |
| `'complete'` | Maximum distance between clusters |
| `'average'` | Average distance between clusters |
| `'single'` | Minimum distance between clusters |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    group_labels: np.ndarray
) -> HBACDetector
```

Fit the HBAC detector.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features) |
| `group_labels` | ndarray | Shape (n_samples,) |

**Returns:** self

### predict()

```python
def predict(embeddings: np.ndarray) -> np.ndarray
```

Predict cluster assignments for new embeddings.

**Returns:** Cluster labels array

### get_dendrogram()

```python
def get_dendrogram() -> dict
```

Get dendrogram data for visualization.

**Returns:** Dictionary with linkage matrix and labels

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `purity_` | float | Cluster purity score (0-1) |
| `linearity_` | float | Linear separability score (0-1) |
| `shortcut_detected_` | bool | Whether shortcut was detected |
| `cluster_labels_` | ndarray | Cluster assignments |
| `dendrogram_` | dict | Dendrogram data |
| `n_clusters_` | int | Number of clusters found |

## Usage Examples

### Basic Usage

```python
from shortcut_detect import HBACDetector

detector = HBACDetector()
detector.fit(embeddings, group_labels)

print(f"Purity: {detector.purity_:.2f}")
print(f"Shortcut detected: {detector.shortcut_detected_}")
```

### Custom Configuration

```python
detector = HBACDetector(
    max_iterations=5,
    min_cluster_size=0.03,
    linkage='complete',
    random_state=42
)
```

### Visualization

```python
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

detector.fit(embeddings, group_labels)

fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(detector.dendrogram_, ax=ax)
plt.title(f"HBAC Dendrogram (Purity: {detector.purity_:.2f})")
plt.savefig("dendrogram.png")
```

## See Also

- [HBAC Method Guide](../methods/hbac.md)
- [ShortcutDetector API](shortcut-detector.md)
