# Advanced analysis

Model comparison, intersectional attributes, custom probes, debiasing experiments, custom tests, bootstraps and batch runs. Snippets that need your own files say so.

## Model comparison

```python
import numpy as np
import pandas as pd
from shortcut_detect import ShortcutDetector

models = {
    "ResNet50": np.load("embeddings_resnet50.npy"),
    "DenseNet121": np.load("embeddings_densenet121.npy"),
    "ViT-B/16": np.load("embeddings_vit.npy"),
}
task_labels = np.load("task_labels.npy")
group_labels = np.load("group_labels.npy")

rows = {}
for name, embeddings in models.items():
    detector = ShortcutDetector(methods=["probe", "statistical", "geometric"], seed=42)
    detector.fit(embeddings, task_labels, group_labels=group_labels)
    res = detector.get_results()
    rows[name] = {
        "probe_score": res["probe"]["results"]["metrics"]["metric_value"],
        "n_significant": sum(len(v or []) for v in res["statistical"]["significant_features"].values()),
        "effect_size": res["geometric"]["bias_pairs"][0].effect_size,
        "risk": res["probe"]["risk_value"],
    }

df = pd.DataFrame(rows).T
print(df)
```

For the paper's measurement per model, call `conditional_audit(embeddings, group_labels, task_labels, config=cfg)` in the same loop and compare `result.results["probe"].statistic` and its confidence interval. `shortcut_detect.comparison.ModelComparisonRunner` runs the detector loop above for the dashboard's comparison tab.

### Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(df.index, df["probe_score"].astype(float))
axes[0].axhline(0.7, color="red", linestyle="--", label="threshold")
axes[0].set_title("probe score")

axes[1].bar(df.index, df["n_significant"].astype(int))
axes[1].set_title("significant dimensions")

axes[2].bar(df.index, df["effect_size"].astype(float))
axes[2].axhline(0.8, color="red", linestyle="--", label="threshold")
axes[2].set_title("geometric effect size")

for ax in axes:
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
```

---

## Intersectional attributes

```python
race = data["race"].to_numpy()
sex = data["sex"].to_numpy()
age_group = data["age_group"].to_numpy()

attributes = {
    "race": race,
    "sex": sex,
    "age": age_group,
    "race_sex": np.array([f"{r}_{s}" for r, s in zip(race, sex)]),
    "race_age": np.array([f"{r}_{a}" for r, a in zip(race, age_group)]),
    "all": np.array([f"{r}_{s}_{a}" for r, s, a in zip(race, sex, age_group)]),
}

for name, labels in attributes.items():
    n_groups = len(np.unique(labels))
    if n_groups < 2:
        continue
    detector = ShortcutDetector(methods=["probe"], seed=42)
    detector.fit(embeddings, task_labels, group_labels=labels)
    score = detector.get_results()["probe"]["results"]["metrics"]["metric_value"]
    print(f"{name:10s} ({n_groups} groups): probe f1 {score:.2f}")
```

The `intersectional` method computes fairness gaps across the intersections directly; see [Intersectional analysis](../methods/intersectional-analysis.md).

---

## Custom probe architectures

```python
import torch.nn as nn
from shortcut_detect import TorchProbe

class AttentionProbe(nn.Module):
    """Probe with self-attention over the embedding."""

    def __init__(self, input_dim, n_classes, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)                       # (batch, 1, dim)
        attn_out, _ = self.attention(x, x, x)
        return self.classifier(attn_out.squeeze(1))

probe = TorchProbe(
    model=AttentionProbe(512, 3),
    loss_fn=nn.CrossEntropyLoss(),
    device="cuda",      # None picks cuda when available, else cpu
    epochs=100,
)
probe.fit(embeddings, group_labels)
print(f"Held-out accuracy: {probe.metric_value_:.2%}")
```

---

## Debiasing experiments

### Linear debiasing

Project the direction between the two group centroids out of the embeddings, then measure again:

```python
from shortcut_detect import GeometricShortcutAnalyzer, ShortcutDetector

analyzer = GeometricShortcutAnalyzer()
analyzer.fit(embeddings, group_labels)
pair = analyzer.bias_pairs_[0]
print(f"Original effect size: {pair.effect_size:.2f}")

g0, g1 = pair.groups
c0, c1 = analyzer.group_stats_[g0]["centroid"], analyzer.group_stats_[g1]["centroid"]
direction = (c1 - c0) / np.linalg.norm(c1 - c0)
embeddings_debiased = embeddings - np.outer(embeddings @ direction, direction)

analyzer_after = GeometricShortcutAnalyzer()
analyzer_after.fit(embeddings_debiased, group_labels)
print("After:", analyzer_after.summary_)   # the centroids now coincide, so bias_pairs_ is empty

def probe_and_count(X):
    detector = ShortcutDetector(methods=["probe", "statistical"], seed=42)
    detector.fit(X, group_labels)
    res = detector.get_results()
    return (res["probe"]["results"]["metrics"]["metric_value"],
            sum(len(v or []) for v in res["statistical"]["significant_features"].values()))

print("before (probe score, significant dims):", probe_and_count(embeddings))
print("after: ", probe_and_count(embeddings_debiased))
```

The package's own mitigations (`AdversarialDebiasing`, `LastLayerRetraining`, `ContrastiveDebiasing` in `shortcut_detect.mitigation`) have their own method pages.

### INLP (iterative null-space projection)

```python
from sklearn.linear_model import LogisticRegression

def inlp_debias(embeddings, group_labels, n_iterations=10):
    X = embeddings.copy()
    for _ in range(n_iterations):
        clf = LogisticRegression(max_iter=1000).fit(X, group_labels)
        w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
        X = X - np.outer(X @ w, w)                     # project the direction out
        if LogisticRegression(max_iter=1000).fit(X, group_labels).score(X, group_labels) < 0.55:
            break
    return X

embeddings_inlp = inlp_debias(embeddings, group_labels)

detector_inlp = ShortcutDetector(methods=["probe"], seed=42)
detector_inlp.fit(embeddings_inlp, group_labels)
print(f"After INLP: probe f1 {detector_inlp.get_results()['probe']['results']['metrics']['metric_value']:.2f}")
```

---

## Custom statistical tests

`GroupDiffTest` calls `test(x1, x2)` on each dimension and reads the p-value from the return value: a float, or a tuple whose second element is the p-value.

```python
from scipy.stats import ks_2samp, f
from shortcut_detect import GroupDiffTest

test_ks = GroupDiffTest(test=ks_2samp)
test_ks.fit(embeddings, group_labels)
ks = test_ks.apply_correction(alpha=0.05, method="fdr_bh", verbose=False)
print("KS:", {k: len(v or []) for k, v in ks["significant_features"].items()})

def variance_ratio_test(group1, group2):
    """Two-sided F test on the variance ratio; returns (statistic, p-value)."""
    ratio = group1.var(ddof=1) / (group2.var(ddof=1) + 1e-10)
    df1, df2 = len(group1) - 1, len(group2) - 1
    p = 2 * min(f.cdf(ratio, df1, df2), 1 - f.cdf(ratio, df1, df2))
    return ratio, p

test_var = GroupDiffTest(test=variance_ratio_test)
test_var.fit(embeddings, group_labels)
var = test_var.apply_correction(alpha=0.05, method="fdr_bh", verbose=False)
print("variance ratio:", {k: len(v or []) for k, v in var["significant_features"].items()})
```

---

## Bootstrapped confidence intervals

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from shortcut_detect import SKLearnProbe

def bootstrap_probe_accuracy(X, y, n_bootstrap=100, seed=0):
    scores = []
    for i in range(n_bootstrap):
        X_boot, y_boot = resample(X, y, random_state=seed + i)
        probe = SKLearnProbe(LogisticRegression(max_iter=1000), metric="accuracy",
                             evaluation="holdout", test_size=0.2, random_state=seed + i)
        probe.fit(X_boot, y_boot)
        scores.append(probe.metric_value_)
    return np.mean(scores), np.percentile(scores, [2.5, 97.5])

mean_acc, (lo, hi) = bootstrap_probe_accuracy(embeddings, group_labels)
print(f"Probe accuracy {mean_acc:.2%} (95% interval {lo:.2%} to {hi:.2%})")
```

The audit computes a bias-corrected bootstrap interval for every detector statistic itself (`ci_low` and `ci_high` on each `DetectorResult`), so the paper's probe needs no extra loop.

---

## Batch processing

```python
from pathlib import Path
import json

def process_dataset(path):
    data = np.load(path)
    detector = ShortcutDetector(methods=["probe", "statistical"], seed=42)
    detector.fit(data["embeddings"], data["groups"])
    res = detector.get_results()
    return {
        "path": str(path),
        "n_samples": int(len(data["embeddings"])),
        "probe_score": float(res["probe"]["results"]["metrics"]["metric_value"]),
        "n_significant": sum(len(v or []) for v in res["statistical"]["significant_features"].values()),
        "risk": res["probe"]["risk_value"],
    }

results = [process_dataset(p) for p in sorted(Path("datasets/").glob("*.npz"))]
with open("batch_results.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(pd.DataFrame(results).to_string())
```

---

## Jupyter notebooks

```bash
jupyter lab examples/
```

- `01_basic_usage/`: unified API, HBAC clustering, statistical testing
- `02_real_data/`: medical imaging demo, embeddings analysis
- `04_risk_recommendations.ipynb`: risk conditions and recommendations

---

## Next steps

- [API reference](../api/shortcut-detector.md): full documentation
- [Detection methods](../methods/overview.md): method details
- [Contributing](../contributing.md): how to contribute
