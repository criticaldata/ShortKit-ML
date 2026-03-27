# Basic Usage Examples

This guide covers fundamental usage patterns for ShortKit-ML.

## Jupyter Notebooks

All examples are available as Jupyter notebooks in the repository:

```bash
cd examples/01_basic_usage/
jupyter lab
```

## Example 1: Unified API Demo

The complete workflow using `ShortcutDetector`.

### Setup

```python
import numpy as np
from shortcut_detect import ShortcutDetector

# Set random seed for reproducibility
np.random.seed(42)
```

### Generate Sample Data

```python
from shortcut_detect import generate_linear_shortcut

# Create data with shortcuts
X, y_task, y_group = generate_linear_shortcut(
    n_samples=1000,
    n_features=512,
    shortcut_strength=0.8,
    random_state=42
)

print(f"Embeddings shape: {X.shape}")
print(f"Groups: {np.unique(y_group)}")
```

### Run Detection

```python
# Create detector with all methods
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric'],
    random_state=42
)

# Fit
detector.fit(X, y_group)

# View summary
print(detector.summary())
```

### Generate Report

```python
# HTML report
detector.generate_report("basic_report.html", format="html")

# View in browser
import webbrowser
webbrowser.open("basic_report.html")
```

---

## Example 2: HBAC Clustering

Standalone HBAC analysis.

```python
from shortcut_detect import HBACDetector
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Create detector
hbac = HBACDetector(
    max_iterations=3,
    min_cluster_size=0.05
)

# Fit
hbac.fit(X, y_group)

# Results
print(f"Purity: {hbac.purity_:.2f}")
print(f"Linearity: {hbac.linearity_:.2f}")
print(f"Shortcut detected: {hbac.shortcut_detected_}")

# Visualize dendrogram
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(hbac.dendrogram_, ax=ax, leaf_rotation=90)
plt.title(f"HBAC Dendrogram (Purity: {hbac.purity_:.2f})")
plt.tight_layout()
plt.savefig("hbac_dendrogram.png", dpi=150)
plt.show()
```

---

## Example 3: Probe-based Detection

Classifier-based information leakage testing.

```python
from shortcut_detect import SKLearnProbe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_group, test_size=0.2, random_state=42
)

# Create and train probe
probe = SKLearnProbe(LogisticRegression(max_iter=1000))
probe.fit(X_train, y_train)

# Evaluate
accuracy = probe.score(X_test, y_test)
print(f"Probe accuracy: {accuracy:.2%}")

# Risk assessment
if accuracy > 0.8:
    print("HIGH RISK: Strong shortcuts detected")
elif accuracy > 0.6:
    print("MEDIUM RISK: Some shortcuts present")
else:
    print("LOW RISK: Minimal shortcuts")
```

### Compare Multiple Classifiers

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': SVC(kernel='linear'),
    'RBF SVM': SVC(kernel='rbf'),
    'Random Forest': RandomForestClassifier(n_estimators=100),
}

print("Classifier Comparison:")
print("-" * 40)
for name, clf in classifiers.items():
    probe = SKLearnProbe(clf)
    probe.fit(X_train, y_train)
    acc = probe.score(X_test, y_test)
    print(f"{name:25s}: {acc:.2%}")
```

---

## Example 4: Statistical Testing

Feature-wise hypothesis testing.

```python
from shortcut_detect import GroupDiffTest
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# Create test
test = GroupDiffTest(
    test=mannwhitneyu,
    alpha=0.05,
    correction='fdr_bh'
)

# Fit
test.fit(X, y_group)

# Results
print(f"Significant features: {test.n_significant_} / {X.shape[1]}")
print(f"Percentage: {test.n_significant_ / X.shape[1]:.1%}")

# Volcano plot
fig, ax = plt.subplots(figsize=(10, 6))
x = test.effect_sizes_
y = -np.log10(test.pvalues_corrected_ + 1e-300)

colors = ['red' if i in test.significant_features_ else 'gray'
          for i in range(len(x))]

ax.scatter(x, y, c=colors, alpha=0.5, s=20)
ax.axhline(-np.log10(0.05), color='blue', linestyle='--',
           label='p=0.05 threshold')
ax.set_xlabel('Effect Size (Cohen\'s d)')
ax.set_ylabel('-log10(corrected p-value)')
ax.set_title(f'Statistical Testing: {test.n_significant_} significant features')
ax.legend()
plt.tight_layout()
plt.savefig("volcano_plot.png", dpi=150)
plt.show()
```

---

## Example 5: Geometric Analysis

Subspace and bias direction analysis.

```python
from shortcut_detect import GeometricShortcutAnalyzer
import matplotlib.pyplot as plt

# Create analyzer
analyzer = GeometricShortcutAnalyzer(n_components=5)
analyzer.fit(X, y_group)

# Results
print(analyzer.summary_)

# Visualize bias direction projections
projections = analyzer.transform(X)

fig, ax = plt.subplots(figsize=(10, 4))
for group in np.unique(y_group):
    mask = y_group == group
    ax.hist(projections[mask], bins=50, alpha=0.5, label=f'Group {group}')

ax.set_xlabel('Projection onto Bias Direction')
ax.set_ylabel('Frequency')
ax.set_title(f'Bias Direction (Effect Size: {analyzer.bias_effect_size_:.2f})')
ax.legend()
plt.tight_layout()
plt.savefig("bias_projections.png", dpi=150)
plt.show()
```

---

## Example 6: Comparison - With vs Without Shortcuts

Compare detection results on clean vs biased data.

```python
from shortcut_detect import (
    ShortcutDetector,
    generate_linear_shortcut,
    generate_no_shortcut
)

# Data WITH shortcuts
X_biased, _, y_biased = generate_linear_shortcut(
    n_samples=500, shortcut_strength=0.9
)

# Data WITHOUT shortcuts
X_clean, _, y_clean = generate_no_shortcut(n_samples=500)

# Detect on both
detector_biased = ShortcutDetector(methods=['probe', 'statistical'])
detector_biased.fit(X_biased, y_biased)

detector_clean = ShortcutDetector(methods=['probe', 'statistical'])
detector_clean.fit(X_clean, y_clean)

# Compare
print("=" * 50)
print("BIASED DATA:")
print(detector_biased.summary())

print("\n" + "=" * 50)
print("CLEAN DATA:")
print(detector_clean.summary())
```

---

## Next Steps

- [Real Data Examples](real-data.md) - CheXpert medical imaging
- [Advanced Analysis](advanced.md) - Model comparison, intersectionality
- [API Reference](../api/shortcut-detector.md) - Full documentation
