# HBACDetector API

`HBACDetector` is an alias of `shortcut_detect.clustering.EmbeddingShortcutDetector`, the Hierarchical Bias-Aware Clustering detector.

## Class reference

::: shortcut_detect.clustering.HBACDetector
    options:
      show_root_heading: true
      show_source: true

## Quick reference

### Constructor

```python
HBACDetector(
    max_iterations: int = 3,
    min_cluster_size: float = 0.01,
    test_size: float = 0.2,
    random_state: int = 42,
    config: HBACConfig | None = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 3 | Number of 2-means splits |
| `min_cluster_size` | float | 0.01 | Smallest cluster kept, as a fraction of the training rows |
| `test_size` | float | 0.2 | Share of rows held out for the linearity test |
| `random_state` | int | 42 | Seed for the split and for k-means |
| `config` | `HBACConfig` | None | Frozen dataclass with the same four fields; takes precedence when given |

```python
from shortcut_detect.clustering import HBACConfig

detector = HBACDetector(config=HBACConfig(max_iterations=5, min_cluster_size=0.03))
```

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> HBACDetector
```

Standardises the embeddings, holds out `test_size` of the rows, runs the 2-means splits, scores purities, ranks dimensions and tests linear separability.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, embedding_dim) |
| `labels` | ndarray | Shape (n_samples,); the group labels |
| `feature_names` | list | Optional names for the dimensions |

**Returns:** self

### get_report()

```python
def get_report() -> dict
```

The standard detector schema: `method`, `shortcut_detected`, `risk_level`, `metrics`, `notes`, `metadata`, `report`.

### get_report_text()

```python
def get_report_text() -> str
```

A multi-line text report with the decision, the cluster purities and the important dimensions.

### summary()

```python
def summary() -> str
```

One line, for example `hbac: shortcut=YES, risk=HIGH, n_clusters=4, linearity_score=1.0, ...`.

### visualize()

```python
def visualize(method: str = "pca", save_path: str | None = None)
```

A two-panel 2-D projection (`"pca"` or `"tsne"`) coloured by cluster and by label.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `shortcut_detected_` | bool | Whether the decision rule fired |
| `shortcut_report_` | dict | `cluster_purities`, `dimension_importance`, `linearity_score`, `statistical_tests`, `has_shortcut` |
| `clusters_` | list | One dict per final cluster with its training row indices and labels |
| `results_` | dict | The standard detector schema |

## Usage examples

### Basic usage

```python
from shortcut_detect import HBACDetector

detector = HBACDetector(random_state=42)
detector.fit(embeddings, group_labels)

verdict = detector.shortcut_report_["has_shortcut"]
print(f"Shortcut detected: {detector.shortcut_detected_} ({verdict['confidence']})")
print(f"Linear test accuracy: {detector.shortcut_report_['linearity_score']['test_accuracy']:.2f}")
```

### Cluster purities

```python
for cluster in detector.shortcut_report_["cluster_purities"]:
    print(cluster["cluster_id"], cluster["size"], round(cluster["purity"], 2), cluster["dominant_label"])
```

### Visualization

```python
detector.fit(embeddings, group_labels)
detector.visualize(method="pca", save_path="hbac_pca.png")
```

## See also

- [HBAC method guide](../methods/hbac.md)
- [ShortcutDetector API](shortcut-detector.md)
