# Probes API

The probes module trains a classifier to predict the group from the embeddings. Both probes subclass `DetectorBase` and do their own train/test split or cross-validation inside `fit`.

## Class reference

### SKLearnProbe

::: shortcut_detect.probes.sklearn_probe.SKLearnProbe
    options:
      show_root_heading: true
      show_source: true

### TorchProbe

::: shortcut_detect.probes.torch_probe.TorchProbe
    options:
      show_root_heading: true
      show_source: true

## SKLearnProbe

### Constructor

```python
SKLearnProbe(
    estimator: BaseEstimator | None = None,
    *,
    metric: str = "f1",
    threshold: float = 0.70,
    average: str = "macro",
    evaluation: str = "holdout",
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | sklearn estimator | standardised `LogisticRegression` | Any estimator with `fit` and `predict` |
| `metric` | str | `"f1"` | `"accuracy"`, `"f1"`, `"precision"`, `"recall"` or `"roc_auc"` |
| `threshold` | float | 0.70 | Shortcut when the metric exceeds it |
| `average` | str | `"macro"` | Averaging for multiclass f1, precision, recall |
| `evaluation` | str | `"holdout"` | `"holdout"`, `"cv"` or `"train"` |
| `test_size` | float | 0.2 | Held-out share for `"holdout"` |
| `cv_folds` | int | 5 | Folds for `"cv"` |
| `random_state` | int | 0 | Seed for the split |

### Methods

#### fit()

```python
def fit(embeddings: np.ndarray, target: np.ndarray) -> SKLearnProbe
```

Splits (or cross-validates), trains the estimator, scores it and fills `results_`.

#### predict()

```python
def predict(X: np.ndarray) -> np.ndarray
```

Labels from the fitted estimator.

#### get_report(), summary()

The standard detector schema and its one-line summary.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `metric_value_` | float | Held-out or cross-validated score |
| `shortcut_detected_` | bool | `metric_value_ > threshold` |
| `estimator_` | estimator | Fitted estimator |
| `y_true_eval_`, `y_pred_eval_` | ndarray | Held-out labels and predictions (holdout and train evaluation) |
| `results_` | dict | Standard schema; `results_["report"]` holds `fold_scores`, `mean_score`, `std_score` under `evaluation="cv"` |

### Usage

```python
from sklearn.linear_model import LogisticRegression
from shortcut_detect import SKLearnProbe

probe = SKLearnProbe(LogisticRegression(max_iter=1000), metric="accuracy",
                     evaluation="holdout", test_size=0.2, random_state=42)
probe.fit(embeddings, group_labels)

print(f"Accuracy: {probe.metric_value_:.2%}, flagged: {probe.shortcut_detected_}")
```

---

## TorchProbe

### Constructor

```python
TorchProbe(
    model: nn.Module,
    loss_fn,
    *,
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,   # {"lr": 1e-3}
    device: str | None = None,
    metric: str = "accuracy",
    threshold: float = 0.70,
    test_size: float = 0.2,
    random_state: int = 0,
    epochs: int = 10,
    batch_size: int = 128,
    num_workers: int = 0,
    early_stopping: int | None = None,
    use_amp: bool = False,
    verbose: bool = False,
    loader_factory=None,
    stage_loader_overrides=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | required | The probe network |
| `loss_fn` | callable | required | For example `nn.CrossEntropyLoss()` |
| `optimizer_class` | class | `torch.optim.Adam` | Optimizer |
| `optimizer_kwargs` | dict | `{"lr": 1e-3}` | Optimizer arguments |
| `device` | str or None | None | `"cuda"` when available, else `"cpu"` |
| `metric` | str | `"accuracy"` | Score reported in `metric_value_` |
| `threshold` | float | 0.70 | Shortcut when the metric exceeds it |
| `test_size` | float | 0.2 | Held-out share |
| `epochs` | int | 10 | Training epochs |
| `batch_size` | int | 128 | Batch size |
| `early_stopping` | int or None | None | Patience in epochs |
| `loader_factory` | callable or None | None | Builds a DataLoader per stage |
| `stage_loader_overrides` | dict or None | None | Per-stage DataLoader kwargs |

### Methods

- `fit(embeddings, target)`: in-memory arrays
- `fit_dataset(dataset, ...)`: a map-style or iterable `torch.utils.data.Dataset` yielding `{"embeddings": x, "labels": y}`
- `fit_loaders(...)`: pre-built DataLoaders; see the class reference
- `predict(X)`: labels from the trained network

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `metric_value_` | float | Held-out score |
| `shortcut_detected_` | bool | `metric_value_ > threshold` |
| `n_features_in_` | int | Input width seen in `fit` |
| `results_` | dict | Standard schema |

### Usage

```python
import torch.nn as nn
from shortcut_detect import TorchProbe

class CustomProbe(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)

probe = TorchProbe(
    model=CustomProbe(512, 3),
    loss_fn=nn.CrossEntropyLoss(),
    device="cuda",
    epochs=50,
)
probe.fit(embeddings, group_labels)
print(probe.metric_value_, probe.shortcut_detected_)
```

## See also

- [Probe method guide](../methods/probe.md)
- [ShortcutDetector API](shortcut-detector.md)
