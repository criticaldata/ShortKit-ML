# Equalized Odds API Reference

Equalized Odds detects shortcuts by measuring **True Positive Rate (TPR) and False Positive Rate (FPR) disparities across protected groups**.
Based on Hardt et al. (2016), this method checks whether predictions satisfy fairness constraints across demographic groups.

This page documents the **standalone EqualizedOddsDetector** and its integration with the unified `ShortcutDetector` API.

---

## `EqualizedOddsDetector`

```python
shortcut_detect.fairness.EqualizedOddsDetector
```

### Description

`EqualizedOddsDetector` trains a lightweight classifier on embeddings and computes per-group TPR and FPR metrics.
Large gaps between groups indicate that embeddings encode shortcuts correlated with protected attributes.

---

## Constructor

```python
EqualizedOddsDetector(
    estimator: Optional[LogisticRegression] = None,
    min_group_size: int = 10,
    tpr_gap_threshold: float = 0.1,
    fpr_gap_threshold: float = 0.1
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `estimator` | Estimator, optional | LogisticRegression(max_iter=1000) | Classifier to train on embeddings |
| `min_group_size` | int | 10 | Minimum samples per group; smaller groups get NaN metrics |
| `tpr_gap_threshold` | float | 0.1 | Threshold for flagging TPR disparity |
| `fpr_gap_threshold` | float | 0.1 | Threshold for flagging FPR disparity |

---

## Methods

### `fit`

```python
fit(
    embeddings: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray
) -> EqualizedOddsDetector
```

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `embeddings` | ndarray `(n_samples, d)` | Precomputed embeddings |
| `labels` | ndarray `(n_samples,)` | Binary task labels (required) |
| `group_labels` | ndarray `(n_samples,)` | Group membership (required) |

#### Returns

* `self`

#### Raises

* `ValueError` if `group_labels` is not provided
* `ValueError` if `labels` is not binary
* `ValueError` if shapes don't align

---

## Attributes (after `fit`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `group_metrics_` | dict | Per-group TPR, FPR, support, and confusion matrix counts |
| `tpr_gap_` | float | Max TPR - Min TPR across groups |
| `fpr_gap_` | float | Max FPR - Min FPR across groups |
| `overall_accuracy_` | float | Classifier accuracy on all samples |
| `report_` | EqualizedOddsReport | Structured report object |

---

## `EqualizedOddsReport`

```python
shortcut_detect.fairness.EqualizedOddsReport
```

Dataclass containing the analysis results.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `group_metrics` | dict | Per-group metrics (TPR, FPR, support, tp, fp, tn, fn) |
| `tpr_gap` | float | Maximum TPR gap across groups |
| `fpr_gap` | float | Maximum FPR gap across groups |
| `overall_accuracy` | float | Overall classifier accuracy |
| `risk_level` | str | "low", "moderate", or "high" |
| `notes` | str | Human-readable interpretation |
| `reference` | str | "Hardt et al. 2016" |

---

## Group Metrics Structure

For each group in `group_metrics`:

```python
{
    "group_id": {
        "tpr": 0.85,      # True Positive Rate
        "fpr": 0.12,      # False Positive Rate
        "support": 150,   # Number of samples
        "tp": 85,         # True Positives
        "fp": 12,         # False Positives
        "tn": 88,         # True Negatives
        "fn": 15          # False Negatives
    }
}
```

---

## Risk Assessment

| Risk Level | Condition | Description |
|------------|-----------|-------------|
| **Low** | Both gaps < threshold | Equalized odds approximately satisfied |
| **Moderate** | Any gap >= threshold | Noticeable disparity |
| **High** | Any gap >= 2x threshold | Large disparity, strong shortcut evidence |

Default thresholds: `tpr_gap_threshold=0.1`, `fpr_gap_threshold=0.1`

---

## Unified API Integration

Equalized Odds is also accessible via `ShortcutDetector`:

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["equalized_odds"])
detector.fit(embeddings, labels, group_labels=groups)

results = detector.get_results()["equalized_odds"]
print(results["tpr_gap"])
print(results["fpr_gap"])
print(results["risk_level"])
```

All reports and visualizations are automatically integrated into:

* `detector.summary()`
* HTML / PDF / Markdown reports
* CSV export

---

## Example

```python
from shortcut_detect.fairness import EqualizedOddsDetector
import numpy as np

# Sample data
np.random.seed(42)
embeddings = np.random.randn(500, 20)
labels = (embeddings[:, 0] > 0).astype(int)
groups = np.array([0] * 250 + [1] * 250)

# Fit detector
detector = EqualizedOddsDetector(
    tpr_gap_threshold=0.1,
    fpr_gap_threshold=0.1
)
detector.fit(embeddings, labels, groups)

# Access results
print(f"TPR Gap: {detector.tpr_gap_:.3f}")
print(f"FPR Gap: {detector.fpr_gap_:.3f}")
print(f"Risk: {detector.report_.risk_level}")

# Per-group metrics
for group, metrics in detector.group_metrics_.items():
    print(f"Group {group}: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
```

---

## Notes and Caveats

* **Binary labels are required**. Multi-class is not supported.
* **Group labels are required**. Cannot operate without them.
* Small groups (< `min_group_size`) will have NaN metrics.
* Measures correlation, not causation—use alongside other methods.

---

## See Also

* [Equalized Odds Overview](../methods/equalized-odds.md)
* [GroupDRO API](groupdro.md)
* [Probe API](probes.md)
* [Statistical Tests API](statistical.md)
* [Unified Detector API](shortcut-detector.md)
