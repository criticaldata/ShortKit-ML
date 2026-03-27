# Bias Direction PCA Detector API

The `BiasDirectionPCADetector` extracts a bias direction from group prototypes using PCA and measures the projection gap across groups to detect systematic embedding bias.

## Class Reference

::: shortcut_detect.geometric.BiasDirectionPCADetector
    options:
      show_root_heading: true
      show_source: true

## Quick Reference

### Constructor

```python
BiasDirectionPCADetector()
```

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    group_labels: np.ndarray,
) -> BiasDirectionPCADetector
```

Fit the bias direction PCA detector on embeddings and group labels.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features), 2D embedding array |
| `group_labels` | ndarray | Shape (n_samples,), group membership labels |

**Returns:** self

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `report_` | BiasDirectionPCAReport | Report with projection gap, explained variance, and per-group projections |

## Usage Examples

### Basic Usage

```python
from shortcut_detect.geometric import BiasDirectionPCADetector

detector = BiasDirectionPCADetector()
detector.fit(embeddings, group_labels)
print(detector.report_)
```

### Via Unified ShortcutDetector

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["bias_direction_pca"])
detector.fit(embeddings, labels, group_labels=group_labels)
print(detector.summary())
```

## See Also

- [Bias Direction PCA Method Guide](../methods/bias-direction-pca.md)
- [Geometric Analyzer API](geometric.md)
- [ShortcutDetector API](shortcut-detector.md)
