# GCE Detector API

The `GCEDetector` class identifies minority/bias-conflicting samples by training a linear classifier with Generalized Cross Entropy loss and flagging high-loss samples.

## Class Reference

::: shortcut_detect.gce.GCEDetector
    options:
      show_root_heading: true
      show_source: true

## Quick Reference

### Constructor

```python
GCEDetector(
    q: float = 0.7,
    loss_percentile_threshold: float = 90.0,
    max_iter: int = 500,
    random_state: int | None = 42,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | float | 0.7 | GCE parameter in (0, 1] |
| `loss_percentile_threshold` | float | 90.0 | Percentile threshold for flagging minority samples |
| `max_iter` | int | 500 | Maximum L-BFGS-B iterations |
| `random_state` | int or None | 42 | Random seed |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> GCEDetector
```

Train a GCE classifier and flag high-loss (minority/bias-conflicting) samples.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features), 2D array |
| `labels` | ndarray | Shape (n_samples,), integer or binary labels |

**Returns:** self

**Raises:**

- `ValueError` if embeddings is not 2D or labels is not 1D
- `ValueError` if embeddings and labels have different lengths
- `ValueError` if fewer than 2 distinct labels

### predict()

```python
def predict(embeddings: np.ndarray) -> np.ndarray
```

Predict class labels from embeddings using the fitted linear classifier.

### get_minority_indices()

```python
def get_minority_indices() -> np.ndarray
```

Return indices of samples flagged as minority/bias-conflicting.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | ndarray | Fitted weight matrix (n_features, n_classes) |
| `intercept_` | ndarray | Fitted bias vector (n_classes,) |
| `classes_` | ndarray | Unique class labels |
| `per_sample_losses_` | ndarray | Per-sample GCE losses |
| `is_minority_` | ndarray | Boolean mask of flagged samples |
| `loss_threshold_` | float | Computed loss threshold |
| `report_` | GCEDetectorReport | Detailed report dataclass |

## Usage Examples

### Basic Usage

```python
from shortcut_detect.gce import GCEDetector

detector = GCEDetector()
detector.fit(embeddings, labels)
print(detector.report_.risk_level)
print(detector.report_.n_minority)
```

### Custom Parameters

```python
detector = GCEDetector(
    q=0.5,
    loss_percentile_threshold=85.0,
    max_iter=1000,
)
detector.fit(embeddings, labels)
minority_idx = detector.get_minority_indices()
```

### Via Unified ShortcutDetector

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["gce"])
detector.fit(embeddings, labels)
print(detector.summary())
```

## See Also

- [GCE Method Guide](../methods/gce.md)
- [ShortcutDetector API](shortcut-detector.md)
