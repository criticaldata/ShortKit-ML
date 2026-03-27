# Advanced Analysis

This guide covers advanced usage patterns including model comparison, custom analysis, and debiasing.

## Model Comparison

Compare shortcut detection across multiple models.

```python
import numpy as np
from shortcut_detect import ShortcutDetector

# Load embeddings from different models
models = {
    'ResNet50': np.load('embeddings_resnet50.npy'),
    'DenseNet121': np.load('embeddings_densenet121.npy'),
    'ViT-B/16': np.load('embeddings_vit.npy'),
    'CLIP': np.load('embeddings_clip.npy'),
}

# Same labels for all models
group_labels = np.load('group_labels.npy')

# Compare
results = {}
for name, embeddings in models.items():
    detector = ShortcutDetector(
        methods=['probe', 'statistical', 'geometric'],
        random_state=42
    )
    detector.fit(embeddings, group_labels)

    res = detector.get_results()
    results[name] = {
        'probe_accuracy': res['probe']['accuracy'],
        'n_significant': len(res['statistical']['significant_features'].get('0_vs_1', [])),
        'effect_size': res['geometric']['effect_size'],
    }

# Display comparison
import pandas as pd
df = pd.DataFrame(results).T
print(df)
```

### Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Probe accuracy
axes[0].bar(results.keys(), [r['probe_accuracy'] for r in results.values()])
axes[0].axhline(0.7, color='red', linestyle='--', label='High risk threshold')
axes[0].set_ylabel('Probe Accuracy')
axes[0].set_title('Probe-based Detection')
axes[0].tick_params(axis='x', rotation=45)

# Significant features %
total_features = embeddings.shape[1]
axes[1].bar(results.keys(),
            [r['n_significant']/total_features*100 for r in results.values()])
axes[1].axhline(30, color='red', linestyle='--', label='High risk threshold')
axes[1].set_ylabel('% Significant Features')
axes[1].set_title('Statistical Testing')
axes[1].tick_params(axis='x', rotation=45)

# Effect size
axes[2].bar(results.keys(), [r['effect_size'] for r in results.values()])
axes[2].axhline(0.7, color='red', linestyle='--', label='High risk threshold')
axes[2].set_ylabel('Effect Size')
axes[2].set_title('Geometric Analysis')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
```

---

## Intersectional Analysis

Analyze shortcuts across multiple protected attributes simultaneously.

```python
from itertools import product

# Define protected attributes
race = data['race'].values
sex = data['sex'].values
age_group = data['age_group'].values

# All combinations
attributes = {
    'race': race,
    'sex': sex,
    'age': age_group,
    'race_sex': np.array([f"{r}_{s}" for r, s in zip(race, sex)]),
    'race_age': np.array([f"{r}_{a}" for r, a in zip(race, age_group)]),
    'all': np.array([f"{r}_{s}_{a}" for r, s, a in zip(race, sex, age_group)]),
}

# Analyze each
intersectional_results = {}
for name, labels in attributes.items():
    n_groups = len(np.unique(labels))
    if n_groups < 2:
        continue

    detector = ShortcutDetector(methods=['probe'])
    detector.fit(embeddings, labels)

    res = detector.get_results()
    intersectional_results[name] = {
        'n_groups': n_groups,
        'probe_accuracy': res['probe']['accuracy'],
    }

    print(f"{name:15s} ({n_groups} groups): {res['probe']['accuracy']:.2%}")
```

---

## Custom Probe Architectures

Create specialized probe classifiers.

```python
import torch
import torch.nn as nn
from shortcut_detect import TorchProbe

class AttentionProbe(nn.Module):
    """Probe with self-attention for interpretability."""

    def __init__(self, input_dim, n_classes, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, dim)
        attn_out, attn_weights = self.attention(x, x, x)
        x = attn_out.squeeze(1)  # (batch, dim)
        return self.classifier(x)

# Use custom probe
probe = TorchProbe(
    model=AttentionProbe(512, 3),
    device='cuda',
    epochs=100
)
probe.fit(X_train, y_train)
print(f"Accuracy: {probe.score(X_test, y_test):.2%}")
```

---

## Debiasing Experiments

Test debiasing techniques and measure their effectiveness.

### Linear Debiasing

```python
from shortcut_detect import GeometricShortcutAnalyzer, ShortcutDetector

# Original analysis
analyzer = GeometricShortcutAnalyzer()
analyzer.fit(embeddings, group_labels)
print(f"Original effect size: {analyzer.bias_effect_size_:.2f}")

# Apply debiasing
embeddings_debiased = analyzer.debias(embeddings)

# Re-analyze
analyzer_after = GeometricShortcutAnalyzer()
analyzer_after.fit(embeddings_debiased, group_labels)
print(f"Debiased effect size: {analyzer_after.bias_effect_size_:.2f}")

# Full comparison
detector_before = ShortcutDetector(methods=['probe', 'statistical'])
detector_before.fit(embeddings, group_labels)

detector_after = ShortcutDetector(methods=['probe', 'statistical'])
detector_after.fit(embeddings_debiased, group_labels)

print("\nBefore debiasing:")
print(f"  Probe accuracy: {detector_before.probe_results_['accuracy']:.2%}")
print(f"  Significant features: {detector_before.statistical_results_['n_significant']}")

print("\nAfter debiasing:")
print(f"  Probe accuracy: {detector_after.probe_results_['accuracy']:.2%}")
print(f"  Significant features: {detector_after.statistical_results_['n_significant']}")
```

### INLP (Iterative Null-space Projection)

```python
def inlp_debias(embeddings, group_labels, n_iterations=10):
    """Iterative Null-space Projection for debiasing."""
    from sklearn.linear_model import LogisticRegression

    X = embeddings.copy()

    for i in range(n_iterations):
        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, group_labels)

        # Get direction
        w = clf.coef_[0]
        w = w / np.linalg.norm(w)

        # Project out
        X = X - (X @ w.reshape(-1, 1)) @ w.reshape(1, -1)

        # Check if still predictable
        clf_check = LogisticRegression(max_iter=1000)
        clf_check.fit(X, group_labels)
        acc = clf_check.score(X, group_labels)

        if acc < 0.55:  # Near random
            break

    return X

# Apply INLP
embeddings_inlp = inlp_debias(embeddings, group_labels)

# Evaluate
detector_inlp = ShortcutDetector(methods=['probe'])
detector_inlp.fit(embeddings_inlp, group_labels)
print(f"After INLP: {detector_inlp.probe_results_['accuracy']:.2%}")
```

---

## Custom Statistical Tests

Use domain-specific statistical tests.

```python
from shortcut_detect import GroupDiffTest
from scipy.stats import ks_2samp, anderson_ksamp

# Kolmogorov-Smirnov test
test_ks = GroupDiffTest(
    test=lambda x, y: ks_2samp(x, y),
    correction='fdr_bh'
)
test_ks.fit(embeddings, group_labels)
print(f"KS test: {test_ks.n_significant_} significant features")

# Custom test function
def custom_test(group1, group2):
    """Custom test: compare variance ratio."""
    var_ratio = group1.var() / (group2.var() + 1e-10)
    # Approximate p-value (simplified)
    from scipy.stats import f
    p_value = 2 * min(
        f.cdf(var_ratio, len(group1)-1, len(group2)-1),
        1 - f.cdf(var_ratio, len(group1)-1, len(group2)-1)
    )
    return type('Result', (), {'pvalue': p_value})()

test_custom = GroupDiffTest(test=custom_test)
test_custom.fit(embeddings, group_labels)
print(f"Variance test: {test_custom.n_significant_} significant features")
```

---

## Bootstrapped Confidence Intervals

Get uncertainty estimates for metrics.

```python
from sklearn.utils import resample

def bootstrap_probe_accuracy(X, y, n_bootstrap=100):
    """Bootstrap confidence interval for probe accuracy."""
    accuracies = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, random_state=None)

        # Train/test split
        n_train = int(0.8 * len(X_boot))
        X_train, X_test = X_boot[:n_train], X_boot[n_train:]
        y_train, y_test = y_boot[:n_train], y_boot[n_train:]

        # Fit probe
        from shortcut_detect import SKLearnProbe
        from sklearn.linear_model import LogisticRegression
        probe = SKLearnProbe(LogisticRegression(max_iter=1000))
        probe.fit(X_train, y_train)
        accuracies.append(probe.score(X_test, y_test))

    return np.mean(accuracies), np.percentile(accuracies, [2.5, 97.5])

mean_acc, (ci_low, ci_high) = bootstrap_probe_accuracy(embeddings, group_labels)
print(f"Probe accuracy: {mean_acc:.2%} (95% CI: {ci_low:.2%} - {ci_high:.2%})")
```

---

## Batch Processing

Process multiple datasets efficiently.

```python
from pathlib import Path
import json

def process_dataset(path):
    """Process a single dataset and return results."""
    data = np.load(path)
    embeddings = data['embeddings']
    group_labels = data['groups']

    detector = ShortcutDetector(methods=['probe', 'statistical'])
    detector.fit(embeddings, group_labels)

    res = detector.get_results()
    return {
        'path': str(path),
        'n_samples': len(embeddings),
        'probe_accuracy': float(res['probe']['accuracy']),
        'n_significant': len(res['statistical']['significant_features'].get('0_vs_1', [])),
    }

# Process all datasets
dataset_dir = Path("datasets/")
results = []

for path in dataset_dir.glob("*.npz"):
    print(f"Processing {path.name}...")
    result = process_dataset(path)
    results.append(result)

# Save results
with open("batch_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Summary
df = pd.DataFrame(results)
print(df.to_string())
```

---

## Jupyter Notebooks

Full notebooks available:

```bash
jupyter lab examples/03_advanced/
```

- `01_model_comparison.ipynb` - Compare multiple models
- `02_custom_analysis.ipynb` - Intersectional + custom visualizations

---

## Next Steps

- [API Reference](../api/shortcut-detector.md) - Full documentation
- [Detection Methods](../methods/overview.md) - Method details
- [Contributing](../contributing.md) - How to contribute
