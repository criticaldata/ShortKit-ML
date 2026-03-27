# Quick Start

Get started with shortcut detection in under 5 minutes.

## Basic Usage

### Step 1: Prepare Your Data

You need two arrays:

- **Embeddings**: `(n_samples, embedding_dim)` - your model's representations
- **Group labels**: `(n_samples,)` - protected attribute labels (e.g., race, sex, age group)

```python
import numpy as np

# Example: Load pre-computed embeddings
embeddings = np.load("embeddings.npy")  # Shape: (1000, 512)
group_labels = np.load("groups.npy")    # Shape: (1000,) with values like 0, 1, 2

print(f"Embeddings: {embeddings.shape}")
print(f"Groups: {np.unique(group_labels)}")
```

### Step 2: Run Detection

```python
from shortcut_detect import ShortcutDetector

# Create detector with all methods
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric']
)

# Fit the detector
detector.fit(embeddings, group_labels)

# Get summary
print(detector.summary())
```

### Step 3: Generate Report

```python
# HTML report (recommended)
detector.generate_report("shortcut_report.html", format="html")

# PDF report
detector.generate_report("shortcut_report.pdf", format="pdf")
```

## Using Synthetic Data

For testing, use built-in data generators:

```python
from shortcut_detect import (
    ShortcutDetector,
    generate_linear_shortcut,
    generate_no_shortcut,
)

# Generate data WITH shortcuts
X_shortcut, y_task, y_group = generate_linear_shortcut(
    n_samples=1000,
    n_features=100,
    shortcut_strength=0.9,  # Strong shortcut
    random_state=42
)

# Detect shortcuts
detector = ShortcutDetector(methods=['probe', 'statistical'])
detector.fit(X_shortcut, y_group)
print(detector.summary())
```

**Expected Output:**
```
======================================================================
UNIFIED SHORTCUT DETECTION SUMMARY
======================================================================
HIGH RISK: Multiple methods detected shortcuts
  - Probe accuracy 94.2% (high risk)
  - 45 features show significant group differences
```

Now test with NO shortcuts:

```python
# Generate data WITHOUT shortcuts
X_clean, y_task, y_group = generate_no_shortcut(
    n_samples=1000,
    n_features=100,
    random_state=42
)

detector_clean = ShortcutDetector(methods=['probe', 'statistical'])
detector_clean.fit(X_clean, y_group)
print(detector_clean.summary())
```

**Expected Output:**
```
======================================================================
UNIFIED SHORTCUT DETECTION SUMMARY
======================================================================
LOW RISK: No significant shortcuts detected
  - Probe accuracy 52.3% (low risk)
  - 2 features show significant group differences
```

## Individual Methods

You can also use detection methods individually:

### HBAC Clustering

```python
from shortcut_detect import HBACDetector

detector = HBACDetector(max_iterations=3, min_cluster_size=0.05)
detector.fit(embeddings, group_labels)

print(f"Purity: {detector.purity_:.2f}")
print(f"Shortcut detected: {detector.shortcut_detected_}")
```

### Probe-based Detection

```python
from shortcut_detect import SKLearnProbe
from sklearn.linear_model import LogisticRegression

probe = SKLearnProbe(LogisticRegression(max_iter=1000))
probe.fit(X_train, y_train)
accuracy = probe.score(X_test, y_test)

print(f"Probe accuracy: {accuracy:.2%}")
print(f"Shortcut risk: {'HIGH' if accuracy > 0.7 else 'LOW'}")
```

### Statistical Testing

```python
from shortcut_detect import GroupDiffTest
from scipy.stats import mannwhitneyu

test = GroupDiffTest(test=mannwhitneyu)
test.fit(embeddings, group_labels)

print(f"Significant features: {test.n_significant_}")
print(f"Feature indices: {test.significant_features_}")
```

### Geometric Analysis

```python
from shortcut_detect import GeometricShortcutAnalyzer

analyzer = GeometricShortcutAnalyzer(n_components=5)
analyzer.fit(embeddings, group_labels)

print(analyzer.summary_)
```

## Embedding-Only Mode

For production systems that only expose embeddings (e.g., HuggingFace Inference API):

```python
from shortcut_detect import ShortcutDetector, HuggingFaceEmbeddingSource

# Your raw inputs
texts = ["patient has mild edema", "normal study", ...]
labels = np.array([1, 0, ...])
groups = np.array([0, 1, ...])

# Create embedding source
hf_source = HuggingFaceEmbeddingSource(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    pooling="mean",
    batch_size=32,
)

# Detect shortcuts (embeddings generated automatically)
detector = ShortcutDetector(methods=["probe", "statistical"])
detector.fit(
    embeddings=None,  # Triggers embedding-only mode
    labels=labels,
    group_labels=groups,
    raw_inputs=texts,
    embedding_source=hf_source,
    embedding_cache_path="embeddings.npy",  # Optional caching
)
```

## Interpretation Guide

| Method | Metric | Low Risk | Medium Risk | High Risk |
|--------|--------|----------|-------------|-----------|
| **HBAC** | Purity | < 60% | 60-80% | > 80% |
| **Probe** | Accuracy | < 60% | 60-80% | > 80% |
| **Statistical** | % Significant Features | < 10% | 10-30% | > 30% |
| **Geometric** | Effect Size | < 0.3 | 0.3-0.7 | > 0.7 |

## Next Steps

- [Interactive Dashboard](dashboard.md) - Visual exploration
- [Detection Methods](../methods/overview.md) - Deep dive into each method
- [API Reference](../api/shortcut-detector.md) - Full API documentation
- [Examples](../examples/basic.md) - Jupyter notebook examples
