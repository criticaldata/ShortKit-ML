# Statistical tests API

The statistical module runs a two-sample test on every embedding dimension and corrects for multiple testing.

## Class reference

::: shortcut_detect.statistical.GroupDiffTest
    options:
      show_root_heading: true
      show_source: true

## GroupDiffTest

`GroupDiffTest` is an alias of `FeatureGroupDiffTest`.

### Constructor

```python
GroupDiffTest(test: callable, **test_kwargs)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `test` | callable | Called as `test(x1, x2, **test_kwargs)` on one dimension's values for two groups; returns a p-value or a tuple whose second element is the p-value |
| `**test_kwargs` | | Forwarded to `test` |

### Test options

```python
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp

test = GroupDiffTest(test=mannwhitneyu)
test = GroupDiffTest(test=ttest_ind)
test = GroupDiffTest(test=ks_2samp)
test = GroupDiffTest(test=mannwhitneyu, alternative="greater")
```

### Correction options

`shortcut_detect.statistical.CORRECTION_METHODS` lists them:

| Value | Description |
|-------|-------------|
| `'fdr_bh'` | Benjamini-Hochberg FDR (used by the pipeline) |
| `'fdr_by'` | Benjamini-Yekutieli FDR |
| `'bonferroni'` | Bonferroni correction |
| `'holm'` | Holm step-down |

## Methods

### fit()

```python
def fit(X: np.ndarray, Y: np.ndarray) -> GroupDiffTest
```

Runs the test on every dimension for every comparison: one comparison with two groups, every one-vs-one and one-vs-rest comparison with three or more.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | ndarray | Shape (n_samples, embedding_dim) |
| `Y` | ndarray | Shape (n_samples,); the group labels |

**Returns:** self

### get_pvalues()

```python
def get_pvalues() -> dict
```

Raw p-values: `{comparison: array of length embedding_dim}`.

### apply_correction()

```python
def apply_correction(alpha: float = 0.05, method: str = "fdr_bh", verbose: bool = True) -> dict
```

Adjusts the p-values of each comparison and updates `shortcut_detected_`. Returns a dict with `significant_features` (indices per comparison, or `None`), `corrected_pvalues`, `rejected` (boolean masks), `method` and `alpha`.

### apply_threshold()

```python
def apply_threshold(alpha: float = 0.05, verbose: bool = True) -> dict
```

Raw `p < alpha` without correction. Returns `{comparison: indices or None}`.

### get_report()

```python
def get_report() -> dict
```

The standard detector schema; `report` holds `p_values` and, after a correction, `significant_features`, `alpha` and `correction_method`.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `p_values_` | dict | Raw p-values per comparison |
| `shortcut_detected_` | bool or None | None after `fit`; set by `apply_threshold` or `apply_correction` |
| `results_` | dict | The standard detector schema |

The class stores no effect sizes or test statistics.

## Usage examples

### Basic usage

```python
from scipy.stats import mannwhitneyu
from shortcut_detect import GroupDiffTest

test = GroupDiffTest(test=mannwhitneyu)
test.fit(embeddings, group_labels)
corrected = test.apply_correction(alpha=0.05, method="fdr_bh", verbose=False)

for comparison, significant in corrected["significant_features"].items():
    print(f"{comparison}: {len(significant or [])} significant dimensions")
```

### With a stricter correction

```python
bonferroni = test.apply_correction(alpha=0.01, method="bonferroni", verbose=False)
```

### Several groups

```python
test = GroupDiffTest(test=mannwhitneyu)
test.fit(embeddings, group_labels)   # three or more group values
print(list(test.get_pvalues()))      # 0_vs_1, 0_vs_2, ..., 0_vs_rest, ...
```

### Volcano plot

```python
import numpy as np
import matplotlib.pyplot as plt

comparison = next(iter(corrected["corrected_pvalues"]))
q = corrected["corrected_pvalues"][comparison]
significant = set(corrected["significant_features"][comparison] or [])

g0, g1 = np.unique(group_labels)[:2]
x0, x1 = embeddings[group_labels == g0], embeddings[group_labels == g1]
pooled = np.sqrt((x0.var(axis=0, ddof=1) + x1.var(axis=0, ddof=1)) / 2)
d = (x1.mean(axis=0) - x0.mean(axis=0)) / pooled

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["red" if i in significant else "gray" for i in range(len(d))]
ax.scatter(d, -np.log10(q + 1e-300), c=colors, alpha=0.5)
ax.set_xlabel("Cohen's d")
ax.set_ylabel("-log10(BH-adjusted p)")
plt.savefig("volcano.png")
```

## See also

- [Statistical testing guide](../methods/statistical.md)
- [ShortcutDetector API](shortcut-detector.md)
