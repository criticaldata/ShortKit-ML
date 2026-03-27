# Real Data Examples

This guide demonstrates shortcut detection on real-world medical imaging data.

## CheXpert Dataset

[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) is a large chest X-ray dataset with demographic information. We use pre-computed embeddings from a trained model.

### Loading CheXpert Embeddings

```python
import numpy as np
import pandas as pd

# Load sample data (included with the library)
data = pd.read_csv("data/chexpert_sample.csv")

# Extract embeddings
embedding_cols = [c for c in data.columns if c.startswith('embedding_')]
embeddings = data[embedding_cols].values

# Labels
task_labels = data['pathology'].values  # Disease labels
group_labels = data['race'].values      # Protected attribute

print(f"Samples: {len(data)}")
print(f"Embedding dim: {embeddings.shape[1]}")
print(f"Groups: {np.unique(group_labels)}")
```

### Running Detection

```python
from shortcut_detect import ShortcutDetector

# Create detector
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric'],
    random_state=42
)

# Fit
detector.fit(embeddings, group_labels, task_labels=task_labels)

# Summary
print(detector.summary())
```

### Expected Output

```
======================================================================
UNIFIED SHORTCUT DETECTION SUMMARY
======================================================================
HIGH RISK: Multiple methods detected shortcuts

HBAC Analysis:
  Purity: 0.78
  Linearity: 0.72
  Status: Shortcuts detected

Probe Analysis:
  Accuracy: 83.2%
  Baseline: 33.3% (3 groups)
  Status: High risk

Statistical Testing:
  Significant features: 156 / 512 (30.5%)
  Status: High risk

Geometric Analysis:
  Bias effect size: 0.89
  Subspace overlap: 0.45
  Status: High risk

RECOMMENDATION: Investigate and mitigate shortcuts before deployment
======================================================================
```

---

## Subgroup Analysis

Analyze shortcuts within specific pathology groups.

```python
# Filter to positive pathology cases
positive_mask = task_labels == 1
embeddings_pos = embeddings[positive_mask]
groups_pos = group_labels[positive_mask]

# Detect shortcuts in positive cases only
detector_pos = ShortcutDetector(methods=['probe', 'statistical'])
detector_pos.fit(embeddings_pos, groups_pos)

print("POSITIVE CASES ONLY:")
print(detector_pos.summary())
```

---

## Visualization

### Embedding Space

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce dimensions
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot by group
fig, ax = plt.subplots(figsize=(10, 8))
for group in np.unique(group_labels):
    mask = group_labels == group
    ax.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        label=group,
        alpha=0.6,
        s=20
    )

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('Embedding Space by Race')
ax.legend()
plt.tight_layout()
plt.savefig("chexpert_tsne.png", dpi=150)
plt.show()
```

### By Pathology and Group

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# By pathology
for path in np.unique(task_labels):
    mask = task_labels == path
    axes[0].scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        label=f'Pathology {path}',
        alpha=0.5,
        s=20
    )
axes[0].set_title('By Pathology')
axes[0].legend()

# By group
for group in np.unique(group_labels):
    mask = group_labels == group
    axes[1].scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        label=group,
        alpha=0.5,
        s=20
    )
axes[1].set_title('By Race')
axes[1].legend()

plt.tight_layout()
plt.savefig("chexpert_comparison.png", dpi=150)
plt.show()
```

---

## Report Generation

### HTML Report

```python
detector.generate_report(
    output_path="chexpert_report.html",
    format="html",
    include_visualizations=True
)

print("Report saved to chexpert_report.html")
```

### PDF Report

```python
detector.generate_report(
    output_path="chexpert_report.pdf",
    format="pdf"
)
```

### CSV Export

```python
from shortcut_detect.reporting import CSVExporter

exporter = CSVExporter(output_dir="./chexpert_results")
files = exporter.export_all(detector)

print(f"Exported files: {files}")
```

---

## Intersectional Analysis

Analyze shortcuts across multiple protected attributes.

```python
# Create intersectional groups
intersectional = data['race'] + '_' + data['sex']
intersectional_labels = intersectional.values

# Detect shortcuts
detector_intersect = ShortcutDetector(methods=['probe', 'statistical'])
detector_intersect.fit(embeddings, intersectional_labels)

print("INTERSECTIONAL ANALYSIS (Race x Sex):")
print(detector_intersect.summary())
```

---

## Temporal Analysis

If your data has timestamps, analyze shortcuts over time.

```python
# Assume 'date' column exists
dates = pd.to_datetime(data['date'])

# Split by year
for year in dates.dt.year.unique():
    year_mask = dates.dt.year == year
    X_year = embeddings[year_mask]
    y_year = group_labels[year_mask]

    if len(np.unique(y_year)) < 2:
        continue

    detector_year = ShortcutDetector(methods=['probe'])
    detector_year.fit(X_year, y_year)

    print(f"Year {year}: Probe accuracy = {detector_year.probe_results_['accuracy']:.2%}")
```

---

## Using the Dashboard

For interactive exploration:

```bash
# Launch dashboard with CheXpert sample data
python app.py
```

1. Go to [http://127.0.0.1:7860](http://127.0.0.1:7860)
2. Click "Load Sample Data" (CheXpert)
3. Select detection methods
4. Click "Run Analysis"
5. Download HTML/PDF report

---

## Jupyter Notebooks

Full notebooks available:

```bash
jupyter lab examples/02_real_data/
```

- `medical_imaging_demo.ipynb` - CheXpert validation (234 samples)
- `embeddings_analysis.ipynb` - Full analysis (2000 samples)

---

## Next Steps

- [Advanced Analysis](advanced.md) - Model comparison, custom analysis
- [API Reference](../api/shortcut-detector.md) - Full documentation
- [Detection Methods](../methods/overview.md) - Method details
