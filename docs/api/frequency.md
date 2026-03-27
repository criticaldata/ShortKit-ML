# FrequencyDetector API

The `FrequencyDetector` class detects embedding-space shortcut signatures by identifying classes whose signal concentrates in a small set of dimensions.

## Class Reference

::: shortcut_detect.frequency.FrequencyDetector
    options:
      show_root_heading: true
      show_source: true

## Quick Reference

### Constructor

```python
FrequencyDetector(
    top_percent: float = 0.05,
    tpr_threshold: float = 0.5,
    fpr_threshold: float = 0.15,
    probe_estimator: Optional[BaseEstimator] = None,
    probe_evaluation: str = "train",
    probe_holdout_frac: float = 0.2,
    random_state: int = 42,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_percent` | float | 0.05 | Fraction of top dimensions to examine |
| `tpr_threshold` | float | 0.5 | Per-class TPR threshold for flagging |
| `fpr_threshold` | float | 0.15 | Per-class FPR threshold for flagging |
| `probe_estimator` | BaseEstimator | None | sklearn classifier (default: LogisticRegression) |
| `probe_evaluation` | str | "train" | "train" or "holdout" |
| `probe_holdout_frac` | float | 0.2 | Holdout fraction for evaluation |
| `random_state` | int | 42 | Random seed |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> FrequencyDetector
```

Fit the frequency detector on embeddings and labels.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features), 2D array |
| `labels` | ndarray | Shape (n_samples,), 1D class labels |

**Returns:** self

**Raises:**

- `ValueError` if embeddings is not 2D or labels is not 1D
- `ValueError` if fewer than 10 samples or fewer than 2 unique classes

### get_report()

```python
def get_report() -> dict
```

Get the detection report after fitting.

**Returns:** Dictionary with `method`, `shortcut_detected`, `risk_level`, `metrics`, `report`, `notes`, and `metadata`.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | FrequencyConfig | Frozen configuration dataclass |
| `probe_` | BaseEstimator | Fitted probe classifier |
| `_is_fitted` | bool | Whether the detector has been fitted |

## Usage Examples

### Basic Usage

```python
from shortcut_detect import FrequencyDetector

detector = FrequencyDetector()
detector.fit(embeddings, labels)
report = detector.get_report()
print(report["shortcut_detected"])
```

### Holdout Evaluation

```python
detector = FrequencyDetector(
    probe_evaluation="holdout",
    probe_holdout_frac=0.2,
    random_state=42,
)
detector.fit(embeddings, labels)
```

### Custom Probe Estimator

```python
from sklearn.svm import LinearSVC

detector = FrequencyDetector(
    probe_estimator=LinearSVC(max_iter=5000),
    top_percent=0.1,
)
detector.fit(embeddings, labels)
```

### Via Unified ShortcutDetector

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(
    methods=["frequency"],
    freq_top_percent=0.05,
    freq_probe_evaluation="holdout",
)
detector.fit(embeddings, labels)
print(detector.summary())
```

## See Also

- [Frequency Method Guide](../methods/frequency.md)
- [ShortcutDetector API](shortcut-detector.md)
