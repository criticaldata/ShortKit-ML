# Geometric Analyzer API

The geometric module analyzes embedding geometry to detect shortcuts.

## Class Reference

::: shortcut_detect.geometric.GeometricShortcutAnalyzer
    options:
      show_root_heading: true
      show_source: true

## GeometricShortcutAnalyzer

### Constructor

```python
GeometricShortcutAnalyzer(
    n_components: int = 5,
    normalize: bool = True,
    random_state: int = None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 5 | PCA components per group |
| `normalize` | bool | True | Normalize embeddings |
| `random_state` | int | None | Random seed |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    group_labels: np.ndarray
) -> GeometricShortcutAnalyzer
```

Analyze geometric structure of embeddings.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features) |
| `group_labels` | ndarray | Shape (n_samples,) |

**Returns:** self

### transform()

```python
def transform(embeddings: np.ndarray) -> np.ndarray
```

Project embeddings onto bias direction.

### debias()

```python
def debias(embeddings: np.ndarray) -> np.ndarray
```

Remove bias direction from embeddings.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `bias_direction_` | ndarray | Unit vector between group centroids |
| `bias_effect_size_` | float | Cohen's d along bias direction |
| `subspace_overlap_` | float | Principal angle overlap (0-1) |
| `group_centroids_` | dict | Centroid per group |
| `group_pca_` | dict | PCA model per group |
| `projections_` | dict | Projections per group |
| `summary_` | str | Human-readable summary |

## Usage Examples

### Basic Usage

```python
from shortcut_detect import GeometricShortcutAnalyzer

analyzer = GeometricShortcutAnalyzer(n_components=5)
analyzer.fit(embeddings, group_labels)

print(analyzer.summary_)
print(f"Effect size: {analyzer.bias_effect_size_:.2f}")
print(f"Subspace overlap: {analyzer.subspace_overlap_:.2f}")
```

### Debiasing

```python
analyzer.fit(embeddings, group_labels)

# Remove bias direction
embeddings_debiased = analyzer.debias(embeddings)

# Verify debiasing
analyzer_after = GeometricShortcutAnalyzer()
analyzer_after.fit(embeddings_debiased, group_labels)
print(f"Effect size after: {analyzer_after.bias_effect_size_:.2f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Project onto bias direction
projections = analyzer.transform(embeddings)

fig, ax = plt.subplots(figsize=(10, 4))
for group in np.unique(group_labels):
    mask = group_labels == group
    ax.hist(projections[mask], bins=50, alpha=0.5, label=f'Group {group}')
ax.legend()
ax.set_xlabel('Bias Direction Projection')
plt.savefig('bias_projections.png')
```

## See Also

- [Geometric Analysis Guide](../methods/geometric.md)
- [ShortcutDetector API](shortcut-detector.md)
