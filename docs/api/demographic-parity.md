# Demographic Parity Detector API

The `DemographicParityDetector` computes the demographic parity gap across demographic groups to detect shortcut reliance on protected attributes.

## Class Reference

::: shortcut_detect.fairness.DemographicParityDetector
    options:
      show_root_heading: true
      show_source: true

## Quick Reference

### Constructor

```python
DemographicParityDetector(
    estimator: LogisticRegression | None = None,
    min_group_size: int = 10,
    dp_gap_threshold: float = 0.1,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | LogisticRegression | None | sklearn classifier (default: LogisticRegression with max_iter=1000) |
| `min_group_size` | int | 10 | Minimum group size; smaller groups get NaN rates |
| `dp_gap_threshold` | float | 0.1 | DP gap threshold for shortcut flagging |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray,
) -> DemographicParityDetector
```

Train classifier and compute demographic parity gap.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features), 2D array |
| `labels` | ndarray | Shape (n_samples,), binary labels |
| `group_labels` | ndarray | Shape (n_samples,), demographic group labels |

**Returns:** self

**Raises:**

- `ValueError` if embeddings is not 2D, labels is not 1D, or group_labels is not 1D
- `ValueError` if shapes do not align
- `ValueError` if labels are not binary (exactly 2 unique values)
- `ValueError` if group_labels is None

### get_report()

```python
def get_report() -> dict[str, Any]
```

Get the detection report after fitting. Inherits from `DetectorBase`.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `group_rates_` | dict | Per-group positive rate and support |
| `dp_gap_` | float | Computed demographic parity gap |
| `overall_positive_rate_` | float | Overall positive prediction rate |
| `report_` | DemographicParityReport | Detailed report dataclass |
| `shortcut_detected_` | bool or None | Whether a shortcut was detected |

## Usage Examples

### Basic Usage

```python
from shortcut_detect.fairness import DemographicParityDetector

detector = DemographicParityDetector()
detector.fit(embeddings, labels, group_labels=group_labels)
report = detector.get_report()
print(report["shortcut_detected"])
print(report["metrics"]["dp_gap"])
```

### Custom Threshold

```python
detector = DemographicParityDetector(
    dp_gap_threshold=0.05,
    min_group_size=20,
)
detector.fit(embeddings, labels, group_labels=group_labels)
```

### Via Unified ShortcutDetector

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["demographic_parity"])
detector.fit(embeddings, labels, group_labels=group_labels)
print(detector.summary())
```

## See Also

- [Demographic Parity Method Guide](../methods/demographic-parity.md)
- [ShortcutDetector API](shortcut-detector.md)
