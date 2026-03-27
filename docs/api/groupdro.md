# GroupDRO API Reference

Group Distributionally Robust Optimization (GroupDRO) detects shortcuts by training a classifier on embeddings using a **worst-group robust objective**.
Persistent performance gaps across groups indicate shortcut reliance or group-specific bias.

This page documents the **standalone GroupDRO detector** and its integration with the unified `ShortcutDetector` API.

---

## `GroupDRODetector`

```python
shortcut_detect.groupdro.GroupDRODetector
```

### Description

`GroupDRODetector` trains a lightweight classifier on embeddings while adversarially upweighting high-loss groups.
It exposes **group-wise accuracy, worst-group accuracy, and adversarial weights** for shortcut detection and analysis.

---

## Constructor

```python
GroupDRODetector(config: Optional[GroupDROConfig] = None)
```

### Parameters

| Name     | Type                       | Description                                              |
| -------- | -------------------------- | -------------------------------------------------------- |
| `config` | `GroupDROConfig`, optional | Configuration object controlling training and robustness |

---

## Methods

### `fit`

```python
fit(
    embeddings: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray
) -> GroupDRODetector
```

#### Parameters

| Name           | Type                     | Description                 |
| -------------- | ------------------------ | --------------------------- |
| `embeddings`   | ndarray `(n_samples, d)` | Precomputed embeddings      |
| `labels`       | ndarray `(n_samples,)`   | Task labels                 |
| `group_labels` | ndarray `(n_samples,)`   | Group membership (required) |

#### Returns

* `self`

#### Raises

* `ValueError` if `group_labels` is not provided

---

### `predict`

```python
predict(embeddings: np.ndarray) -> np.ndarray
```

Predict task labels using the trained classifier.

---

### `get_report`

```python
get_report() -> Dict[str, Any]
```

Returns the standardized detector report (`results_`). The detailed GroupDRO training report is
available under `report = detector.get_report()["report"]`.

---

## `GroupDROConfig`

```python
shortcut_detect.groupdro.GroupDROConfig
```

Configuration dataclass controlling training and robustness behavior.

---

### Training Parameters

| Parameter      | Type  | Default | Description                   |
| -------------- | ----- | ------- | ----------------------------- |
| `n_epochs`     | int   | 10      | Number of training epochs     |
| `batch_size`   | int   | 128     | Batch size                    |
| `lr`           | float | 1e-3    | Learning rate                 |
| `weight_decay` | float | 5e-5    | L2 regularization             |
| `momentum`     | float | 0.9     | SGD momentum                  |
| `num_workers`  | int   | 0       | DataLoader workers            |
| `loader_factory` | callable or None | None | Optional hook to build train/val/full loaders |
| `stage_loader_overrides` | dict or None | None | Per-stage DataLoader kwargs overrides |
| `val_fraction` | float | 0.1     | Fraction for validation split |

---

### Robust Optimization Parameters

| Parameter                    | Type         | Default | Description                           |
| ---------------------------- | ------------ | ------- | ------------------------------------- |
| `robust`                     | bool         | True    | Enable GroupDRO objective             |
| `alpha`                      | float        | 0.2     | Mass for BTL/greedy objective         |
| `robust_step_size`           | float        | 0.01    | Adversarial update step size          |
| `gamma`                      | float        | 0.1     | EMA decay for historical group loss   |
| `use_normalized_loss`        | bool         | False   | Normalize group losses before update  |
| `btl`                        | bool         | False   | Use greedy (BTL-style) objective      |
| `minimum_variational_weight` | float        | 0.0     | Weight smoothing in greedy objective  |
| `generalization_adjustment`  | list or None | None    | Per-group adjustment term             |
| `automatic_adjustment`       | bool         | False   | Update adjustment using train–val gap |

---

### Model Parameters

| Parameter    | Type        | Default | Description                |
| ------------ | ----------- | ------- | -------------------------- |
| `hidden_dim` | int or None | None    | Optional hidden layer size |
| `dropout`    | float       | 0.0     | Dropout rate               |

---

### Miscellaneous

| Parameter | Type        | Default | Description         |
| --------- | ----------- | ------- | ------------------- |
| `seed`    | int         | 0       | Random seed         |
| `device`  | str or None | None    | `"cpu"` or `"cuda"` |

---

## Report Dictionary Structure

```python
report = detector.get_report()["report"]
```

### GroupDRO report keys

| Key               | Type    | Description                             |
| ----------------- | ------- | --------------------------------------- |
| `success`         | bool    | Whether training completed successfully |
| `method`          | str     | `"groupdro"`                            |
| `n_groups`        | int     | Number of groups                        |
| `group_id_map`    | dict    | Original group id → internal index      |
| `history`         | list    | Per-epoch metrics                       |
| `final`           | dict    | Final evaluation metrics                |
| `final_adv_probs` | ndarray | Final adversarial group weights         |

---

### `history` entries

Each element in `history` is a dict containing:

| Key                     | Description                     |
| ----------------------- | ------------------------------- |
| `epoch`                 | Epoch index                     |
| `train_avg_acc`         | Average training accuracy       |
| `train_worst_group_acc` | Worst-group training accuracy   |
| `val_avg_acc`           | Average validation accuracy     |
| `val_worst_group_acc`   | Worst-group validation accuracy |

(Additional group-wise stats are also included.)

---

### `final` metrics

| Key                | Description                |
| ------------------ | -------------------------- |
| `avg_acc`          | Overall accuracy           |
| `worst_group_acc`  | Worst-group accuracy       |
| `avg_acc_group:i`  | Accuracy for group *i*     |
| `avg_loss_group:i` | Average loss for group *i* |

---

## Unified API Integration

GroupDRO is also accessible via `ShortcutDetector`:

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["groupdro"])
detector.fit(embeddings, labels, group_labels=groups)

results = detector.get_results()["groupdro"]["report"]
```

All reports and visualizations are automatically integrated into:

* `detector.summary()`
* HTML / PDF / Markdown reports
* CSV export

---

## Notes and Caveats

* **Group labels are required**. GroupDRO cannot operate without them.
* Performance gaps do **not guarantee causality**; use alongside HBAC or probes.
* Small groups (< 20 samples) may produce unstable estimates.
* GroupDRO detects *effects*, not which features cause them.

---

## See Also

* [GroupDRO Overview](../methods/groupdro.md)
* [HBAC API](hbac.md)
* [Probe API](probes.md)
* [Statistical Tests API](statistical.md)
* [Unified Detector API](shortcut-detector.md)
