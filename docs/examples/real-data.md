# Real data examples

This page shows the package on chest X-ray embeddings with demographic metadata.

!!! note "Illustrative code"
    The file paths and printed values on this page illustrate the API; they are not results from the paper. The paper's CheXpert arrays (seven backbones over the first 2,000 images of the official `train.csv`, 477 patients) are on the gated Hugging Face dataset described in [the paper's pipeline](../reproducibility.md#paper-pipeline), and its measurement is `shortcut_detect.audit.conditional_audit`.

## Loading embeddings

The package ships no sample CSV. Two ways to get arrays:

From your own CSV in the dashboard's format (`embedding_0` to `embedding_{d-1}`, `task_label`, `group_label`):

```python
import numpy as np
import pandas as pd

data = pd.read_csv("my_chexpert_embeddings.csv")
embedding_cols = [c for c in data.columns if c.startswith("embedding_")]
embeddings = data[embedding_cols].to_numpy()
task_labels = data["task_label"].to_numpy()    # e.g. pneumothorax 0/1
group_labels = data["group_label"].to_numpy()  # e.g. race

print(f"Samples: {len(data)}, embedding dim: {embeddings.shape[1]}, groups: {np.unique(group_labels)}")
```

From the paper's arrays, once you have access to the hub dataset (`hf download MITCriticalData/ShortKit-ML-data --repo-type dataset --local-dir data/`):

```python
embeddings = np.load("data/chexpert_multibackbone/densenet121_embeddings.npy")
meta = pd.read_csv("data/chexpert_multibackbone/densenet121_metadata.csv")
task_labels = meta["task_label"].to_numpy()
group_labels = meta["sex"].to_numpy()          # the file also has "race" and "age_bin"
```

## The audit

```python
from shortcut_detect.audit import AuditConfig, conditional_audit
from shortcut_detect.audit.report import audit_report, summary_text

cfg = AuditConfig(n_permutations=200, n_permutations_probe=500, seed=42)
result = conditional_audit(embeddings, group_labels, task_labels, config=cfg)   # add groups=patient_ids when patients repeat

print(summary_text(result, attribute="sex", task="the task label"))
audit_report(result, "chexpert_audit.html", title="DenseNet-121, sex", attribute="sex", task="the task label")
```

---

## Running the detector pipeline

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["hbac", "probe", "statistical", "geometric"], seed=42)
detector.fit(embeddings, task_labels, group_labels=group_labels)
print(detector.summary())
```

`summary()` prints one block per method with its risk label and reason, then the overall assessment (`HIGH RISK`, `MODERATE RISK` or `LOW RISK` under the default `indicator_count` condition). The numbers behind it are in `get_results()`:

```python
results = detector.get_results()
print(results["probe"]["risk_value"], results["probe"]["results"]["metrics"])
print(results["hbac"]["report"]["has_shortcut"])
print({k: len(v or []) for k, v in results["statistical"]["significant_features"].items()})
print(results["geometric"]["summary"])
```

---

## Subgroup analysis

Shortcuts within the positive cases only:

```python
positive_mask = task_labels == 1
embeddings_pos = embeddings[positive_mask]
groups_pos = group_labels[positive_mask]

detector_pos = ShortcutDetector(methods=["probe", "statistical"], seed=42)
detector_pos.fit(embeddings_pos, groups_pos)
print(detector_pos.summary())
```

---

## Visualization

### Embedding space

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(10, 8))
for group in np.unique(group_labels):
    mask = group_labels == group
    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=group, alpha=0.6, s=20)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend()
plt.tight_layout()
plt.savefig("chexpert_tsne.png", dpi=150)
```

### By task and by group

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for label in np.unique(task_labels):
    mask = task_labels == label
    axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f"task {label}", alpha=0.5, s=20)
axes[0].set_title("By task label")
axes[0].legend()

for group in np.unique(group_labels):
    mask = group_labels == group
    axes[1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=group, alpha=0.5, s=20)
axes[1].set_title("By group")
axes[1].legend()

plt.tight_layout()
plt.savefig("chexpert_comparison.png", dpi=150)
```

---

## Report generation

### HTML, PDF and Markdown

```python
detector.generate_report(output_path="chexpert_report.html", format="html", include_visualizations=True)
detector.generate_report(output_path="chexpert_report.pdf", format="pdf")      # needs the WeasyPrint system libraries
detector.generate_report(output_path="chexpert_report.md", format="markdown")
```

### CSV export

```python
from shortcut_detect.reporting import export_to_csv

files = export_to_csv(detector, "./chexpert_results")
print(files)   # {name: path}
```

---

## Intersectional attributes

With `race` and `sex` columns in the CSV:

```python
intersectional_labels = (data["race"].astype(str) + "_" + data["sex"].astype(str)).to_numpy()

detector_intersect = ShortcutDetector(methods=["probe", "statistical"], seed=42)
detector_intersect.fit(embeddings, task_labels, group_labels=intersectional_labels)
print(detector_intersect.summary())
```

---

## Temporal analysis

With a `date` column in the CSV:

```python
dates = pd.to_datetime(data["date"])

for year in sorted(dates.dt.year.unique()):
    mask = (dates.dt.year == year).to_numpy()
    if len(np.unique(group_labels[mask])) < 2:
        continue
    detector_year = ShortcutDetector(methods=["probe"], seed=42)
    detector_year.fit(embeddings[mask], task_labels[mask], group_labels=group_labels[mask])
    score = detector_year.get_results()["probe"]["results"]["metrics"]["metric_value"]
    print(f"{year}: probe f1 {score:.2f}")
```

---

## Using the dashboard

```bash
python app.py
```

1. Open [http://127.0.0.1:7860](http://127.0.0.1:7860)
2. Keep the sample data or upload a CSV
3. Select detection methods
4. Click **Run Detection**
5. Download the HTML or PDF report

---

## Jupyter notebooks

```bash
jupyter lab examples/02_real_data/
```

- `medical_imaging_demo.ipynb`: CheXpert demo

---

## Next steps

- [Advanced analysis](advanced.md): model comparison, custom analysis
- [API reference](../api/shortcut-detector.md): full documentation
- [Detection methods](../methods/overview.md): method details
