# Geometric analyzer API

The geometric module compares the groups' centroids and principal subspaces.

## Class reference

::: shortcut_detect.geometric.GeometricShortcutAnalyzer
    options:
      show_root_heading: true
      show_source: true

## GeometricShortcutAnalyzer

### Constructor

```python
GeometricShortcutAnalyzer(
    n_components: int = 5,
    min_group_size: int = 20,
    effect_threshold: float = 0.8,
    subspace_cosine_threshold: float = 0.85,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 5 | Principal directions kept per group |
| `min_group_size` | int | 20 | Groups with fewer rows are skipped |
| `effect_threshold` | float | 0.8 | Effect size at or above which a centroid pair counts as a signal |
| `subspace_cosine_threshold` | float | 0.85 | Mean principal cosine at or above which a subspace pair counts as a signal |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    group_labels: Sequence,
) -> GeometricShortcutAnalyzer
```

Computes per-group statistics, every pairwise bias direction and every pairwise subspace comparison, then the risk level.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, embedding_dim) |
| `group_labels` | sequence | Length n_samples; at least two groups of `min_group_size` rows |

**Returns:** self

### get_report()

```python
def get_report() -> dict
```

The standard detector schema; `report` holds `summary`, `bias_pairs` and `subspace_pairs` as plain dicts.

### summary()

```python
def summary() -> str
```

One line with the risk level and the pair counts.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `group_stats_` | dict | Per group, keyed by its label as a string: `count`, `centroid`, `components`, `variance_fraction`, `mask`, `projection_std` |
| `bias_pairs_` | list of `BiasPairStats` | `groups`, `direction_norm`, `effect_size`, `projection_gap`, `alignment_score`; sorted by effect size |
| `subspace_pairs_` | list of `SubspacePairStats` | `groups`, `mean_cosine`, `min_angle_deg`, `max_angle_deg`; sorted by mean cosine |
| `summary_` | dict | `risk_level`, `message`, `num_high_effect_pairs`, `num_overlap_pairs` |
| `shortcut_detected_` | bool | True at moderate or high risk |

## Usage examples

### Basic usage

```python
from shortcut_detect import GeometricShortcutAnalyzer

analyzer = GeometricShortcutAnalyzer(n_components=5)
analyzer.fit(embeddings, group_labels)

print(analyzer.summary_)
print(f"Effect size: {analyzer.bias_pairs_[0].effect_size:.2f}")
print(f"Mean subspace cosine: {analyzer.subspace_pairs_[0].mean_cosine:.2f}")
```

### Projections onto the bias direction

```python
import numpy as np
import matplotlib.pyplot as plt

g0, g1 = analyzer.bias_pairs_[0].groups
c0, c1 = analyzer.group_stats_[g0]["centroid"], analyzer.group_stats_[g1]["centroid"]
direction = (c1 - c0) / np.linalg.norm(c1 - c0)
projections = embeddings @ direction

fig, ax = plt.subplots(figsize=(10, 4))
for group in (g0, g1):
    mask = analyzer.group_stats_[group]["mask"]
    ax.hist(projections[mask], bins=50, alpha=0.5, label=f"group {group}")
ax.legend()
ax.set_xlabel("projection onto the bias direction")
plt.savefig("bias_projections.png")
```

## See also

- [Geometric analysis guide](../methods/geometric.md)
- [ShortcutDetector API](shortcut-detector.md)
