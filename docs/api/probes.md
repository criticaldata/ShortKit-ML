# Probes API

The probes module provides classifier-based shortcut detection.

## Class Reference

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

Probe using scikit-learn classifiers.

### Constructor

```python
SKLearnProbe(
    classifier: sklearn.base.ClassifierMixin = None,
    cv: int = 5
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifier` | ClassifierMixin | LogisticRegression | sklearn classifier |
| `cv` | int | 5 | Cross-validation folds |

### Methods

#### fit()

```python
def fit(X: np.ndarray, y: np.ndarray) -> SKLearnProbe
```

Train the probe classifier.

#### score()

```python
def score(X: np.ndarray, y: np.ndarray) -> float
```

Evaluate accuracy on test data.

#### predict()

```python
def predict(X: np.ndarray) -> np.ndarray
```

Predict group labels.

#### predict_proba()

```python
def predict_proba(X: np.ndarray) -> np.ndarray
```

Predict class probabilities.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `accuracy_` | float | Training accuracy |
| `cv_scores_` | ndarray | Cross-validation scores |
| `classifier` | object | Fitted classifier |

### Usage

```python
from shortcut_detect import SKLearnProbe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, group_labels, test_size=0.2
)

probe = SKLearnProbe(LogisticRegression(max_iter=1000))
probe.fit(X_train, y_train)

accuracy = probe.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

---

## TorchProbe

Probe using PyTorch models with GPU support.

### Constructor

```python
TorchProbe(
    model: torch.nn.Module = None,
    device: str = 'cpu',
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    early_stopping: int = 10
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | MLP | PyTorch model |
| `device` | str | 'cpu' | Device ('cpu' or 'cuda') |
| `epochs` | int | 100 | Training epochs |
| `learning_rate` | float | 1e-3 | Learning rate |
| `batch_size` | int | 64 | Batch size |
| `early_stopping` | int | 10 | Early stopping patience |
| `loader_factory` | callable or None | None | Optional hook to build loaders by stage |
| `stage_loader_overrides` | dict or None | None | Per-stage DataLoader kwargs overrides |

### Methods

Same as SKLearnProbe: `fit()`, `score()`, `predict()`, `predict_proba()`

### Additional Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `train_losses_` | list | Training loss history |
| `val_losses_` | list | Validation loss history |

### Usage

```python
from shortcut_detect import TorchProbe
import torch.nn as nn

class CustomProbe(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)

probe = TorchProbe(
    model=CustomProbe(512, 3),
    device='cuda',
    epochs=50
)
probe.fit(X_train, y_train)
accuracy = probe.score(X_test, y_test)
```

## Base Probe Class

### Probe

Abstract base class for all probes.

```python
from shortcut_detect.probes import Probe

class MyCustomProbe(Probe):
    def fit(self, X, y):
        # Training logic
        return self

    def score(self, X, y):
        # Evaluation logic
        return accuracy

    def predict(self, X):
        # Prediction logic
        return predictions
```

## See Also

- [Probe Method Guide](../methods/probe.md)
- [ShortcutDetector API](shortcut-detector.md)
