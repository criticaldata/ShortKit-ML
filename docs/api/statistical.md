# Statistical Tests API

The statistical module provides feature-wise hypothesis testing for shortcut detection.

## Class Reference

::: shortcut_detect.statistical.GroupDiffTest
    options:
      show_root_heading: true
      show_source: true

## GroupDiffTest

### Constructor

```python
GroupDiffTest(
    test: callable = mannwhitneyu,
    alpha: float = 0.05,
    correction: str = 'fdr_bh',
    alternative: str = 'two-sided'
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test` | callable | `mannwhitneyu` | Statistical test function |
| `alpha` | float | 0.05 | Significance level |
| `correction` | str | 'fdr_bh' | Multiple testing correction |
| `alternative` | str | 'two-sided' | Alternative hypothesis |

### Test Options

```python
from scipy.stats import mannwhitneyu, ttest_ind, kruskal, f_oneway

# Non-parametric (recommended)
test = GroupDiffTest(test=mannwhitneyu)  # 2 groups
test = GroupDiffTest(test=kruskal)       # 3+ groups

# Parametric
test = GroupDiffTest(test=ttest_ind)     # 2 groups
test = GroupDiffTest(test=f_oneway)      # 3+ groups
```

### Correction Options

| Value | Description |
|-------|-------------|
| `'fdr_bh'` | Benjamini-Hochberg FDR (default) |
| `'bonferroni'` | Bonferroni correction |
| `'holm'` | Holm-Bonferroni |
| `'sidak'` | Sidak correction |
| `None` | No correction |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    group_labels: np.ndarray
) -> GroupDiffTest
```

Run statistical tests on all dimensions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features) |
| `group_labels` | ndarray | Shape (n_samples,) |

**Returns:** self

### get_significant_features()

```python
def get_significant_features(alpha: float = None) -> np.ndarray
```

Get indices of significant features at given alpha.

### get_effect_sizes()

```python
def get_effect_sizes() -> np.ndarray
```

Get effect sizes (Cohen's d) for each feature.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `pvalues_` | ndarray | Raw p-values per dimension |
| `pvalues_corrected_` | ndarray | Corrected p-values |
| `significant_features_` | ndarray | Indices of significant features |
| `n_significant_` | int | Number of significant features |
| `effect_sizes_` | ndarray | Effect sizes per dimension |
| `statistics_` | ndarray | Test statistics per dimension |

## Usage Examples

### Basic Usage

```python
from shortcut_detect import GroupDiffTest
from scipy.stats import mannwhitneyu

test = GroupDiffTest(test=mannwhitneyu)
test.fit(embeddings, group_labels)

print(f"Significant features: {test.n_significant_}")
print(f"Feature indices: {test.significant_features_[:10]}")
```

### With Custom Correction

```python
test = GroupDiffTest(
    test=mannwhitneyu,
    alpha=0.01,
    correction='bonferroni'
)
test.fit(embeddings, group_labels)
```

### Multi-group Testing

```python
from scipy.stats import kruskal

test = GroupDiffTest(test=kruskal)
test.fit(embeddings, group_labels)  # Works with 3+ groups
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Volcano plot
fig, ax = plt.subplots(figsize=(10, 6))
x = test.effect_sizes_
y = -np.log10(test.pvalues_corrected_ + 1e-300)
colors = ['red' if i in test.significant_features_ else 'gray'
          for i in range(len(x))]
ax.scatter(x, y, c=colors, alpha=0.5)
ax.set_xlabel('Effect Size')
ax.set_ylabel('-log10(p-value)')
plt.savefig('volcano.png')
```

## See Also

- [Statistical Testing Guide](../methods/statistical.md)
- [ShortcutDetector API](shortcut-detector.md)
