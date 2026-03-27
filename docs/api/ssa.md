# SSA Detector API

The `SSADetector` class implements semi-supervised shortcut detection by pseudo-labeling spurious attributes and running GroupDRO on pseudo-groups.

## Class Reference

::: shortcut_detect.ssa.SSADetector
    options:
      show_root_heading: true
      show_source: true

## Quick Reference

### Constructor

```python
SSADetector(config: SSAConfig | None = None)
```

### SSAConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `K` | int | 3 | Number of K-fold splits |
| `T` | int | 2000 | Training iterations per fold |
| `batch_size` | int | 128 | Batch size |
| `lr` | float | 1e-3 | Learning rate |
| `weight_decay` | float | 1e-4 | Weight decay |
| `momentum` | float | 0.9 | SGD momentum |
| `hidden_dim` | int or None | None | Hidden layer size (None = linear) |
| `dropout` | float | 0.0 | Dropout rate |
| `tau_gmin` | float | 0.95 | Confidence threshold for smallest group |
| `threshold_update_every` | int | 200 | Threshold update frequency |
| `dl_val_fraction` | float | 0.5 | Fraction of DL used for validation |
| `seed` | int | 0 | Random seed |
| `device` | str or None | None | PyTorch device |
| `groupdro` | GroupDROConfig | default | GroupDRO configuration for Phase 2 |
| `ssa_gap_threshold` | float | 0.10 | Accuracy gap threshold for detection |

## Methods

### fit()

```python
def fit(
    du_embeddings: np.ndarray,
    du_labels: np.ndarray,
    dl_embeddings: np.ndarray,
    dl_labels: np.ndarray,
    dl_spurious: np.ndarray,
) -> SSADetector
```

Run end-to-end SSA: Phase 1 pseudo-labeling + Phase 2 GroupDRO.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `du_embeddings` | ndarray | (n_unlabeled, n_features) unlabeled embeddings |
| `du_labels` | ndarray | (n_unlabeled,) task labels for unlabeled data |
| `dl_embeddings` | ndarray | (n_labeled, n_features) labeled embeddings |
| `dl_labels` | ndarray | (n_labeled,) task labels for labeled data |
| `dl_spurious` | ndarray | (n_labeled,) spurious attribute labels for labeled data |

**Returns:** self

### get_report()

```python
def get_report() -> dict
```

Get the detection report after fitting. Inherits from `DetectorBase`.

### get_shortcut_detected()

```python
def get_shortcut_detected(avg_acc: float, worst_acc: float) -> bool | None
```

Detect a shortcut if the worst-group accuracy gap exceeds the threshold.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `a_hat_` | ndarray | Pseudo spurious-attribute labels for DU |
| `groupdro_` | GroupDRODetector | Fitted GroupDRO detector from Phase 2 |
| `attr_models_` | list[nn.Module] | Attribute predictor models (one per fold) |
| `shortcut_detected_` | bool or None | Whether a shortcut was detected |
| `report_` | dict | Detailed report including pseudo labels and GroupDRO results |

## Usage Examples

### Basic Usage

```python
from shortcut_detect.ssa import SSADetector, SSAConfig

detector = SSADetector()
detector.fit(
    du_embeddings=unlabeled_emb,
    du_labels=unlabeled_y,
    dl_embeddings=labeled_emb,
    dl_labels=labeled_y,
    dl_spurious=labeled_attrs,
)
report = detector.get_report()
print(report["shortcut_detected"])
```

### Custom Configuration

```python
config = SSAConfig(
    K=5,
    T=3000,
    tau_gmin=0.90,
    hidden_dim=64,
    dropout=0.1,
    seed=42,
)
detector = SSADetector(config=config)
detector.fit(
    du_embeddings=unlabeled_emb,
    du_labels=unlabeled_y,
    dl_embeddings=labeled_emb,
    dl_labels=labeled_y,
    dl_spurious=labeled_attrs,
)
```

### Via Unified ShortcutDetector

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["ssa"])
detector.fit(
    embeddings=unlabeled_emb,
    labels=unlabeled_y,
    dl_embeddings=labeled_emb,
    dl_labels=labeled_y,
    dl_spurious=labeled_attrs,
)
print(detector.summary())
```

## See Also

- [SSA Method Guide](../methods/ssa.md)
- [GroupDRO API](groupdro.md)
- [ShortcutDetector API](shortcut-detector.md)
