# Quick start

Run the paper's audit in a few minutes, then the other package methods.

## The audit

### Step 1: prepare your data

The audit used in the paper needs four arrays:

- **Embeddings** `E`: `(n_samples, embedding_dim)` - your frozen model's representations
- **Attribute** `A`: `(n_samples,)` - the sensitive attribute (e.g., sex, race, age group)
- **Task label** `Y`: `(n_samples,)` - the binary task label the audit conditions on
- **Patient ids** `groups`: `(n_samples,)` - optional; needed whenever one patient has several images

```python
import numpy as np

E = np.load("embeddings.npy")          # Shape: (1000, 512)
A = np.load("attribute.npy")           # Shape: (1000,), e.g. 0/1 or strings
Y = np.load("task_labels.npy")         # Shape: (1000,), 0/1
patient_ids = np.load("patients.npy")  # Shape: (1000,)
```

The width (512 here) is arbitrary: 2, 384, 4096 or any other size works. These four arrays are all the package needs, whatever model or modality (image, text, audio, tabular, multimodal) produced the embeddings.

### Step 2: run the conditional audit

```python
from shortcut_detect.audit import AuditConfig, conditional_audit

cfg = AuditConfig(n_permutations=200, n_permutations_probe=500, seed=42)
result = conditional_audit(E, A, Y, config=cfg, groups=patient_ids)

probe = result.results["probe"]  # the primary test
print(f"Conditional probe AUROC: {probe.statistic:.3f} "
      f"(95% CI {probe.ci_low:.3f} to {probe.ci_high:.3f}), "
      f"permutation p = {probe.p_value:.3f}, flag = {probe.flag}")
print(f"Detectors flagged (descriptive): {result.n_flagged} of {len(result.results)}")
```

The probe predicts `A` from `[E, Y]`; its score is the out-of-fold AUROC and its p-value comes from shuffling `A` within strata of `Y`, between patients when `groups` is given. The five secondary detectors (`statistical`, `geometric`, `bias_direction`, `hbac_nmi`, `frequency_topk`) share that null and are reported as diagnostics. A flagged probe means the attribute is recoverable from the embedding given the task; whether a classifier uses it is a separate question. See the [Audit API](../api/audit.md).

### Step 3: write the report

`audit_report` writes the audit as a self-contained HTML page or a PDF; the file extension picks the format. `summary_text` returns the one-paragraph verdict the report opens with.

```python
from shortcut_detect.audit.report import audit_report, summary_text

print(summary_text(result, attribute="sex", task="pneumothorax"))

audit_report(result, "audit.html", title="ViT-B/16, sex given pneumothorax",
             attribute="sex", task="pneumothorax")
audit_report(result, "audit.pdf", title="ViT-B/16, sex given pneumothorax",
             attribute="sex", task="pneumothorax")
```

The report holds the summary paragraph, the setting (sample size, groups, permutations, alpha), one row per detector (statistic, 95% CI, null mean and SD, p, BH q, flag) and a figure of each statistic against its null. `result.rows()` returns the same table as a list of dicts, ready for a DataFrame or a CSV.

## Other package methods

The `ShortcutDetector` pipeline predates the audit. It runs any of the package's 19 registered methods (`shortcut_detect.discovery.available_method_plugins()`), compares each against a fixed threshold and aggregates the flags into a LOW, MODERATE or HIGH label. It drives the dashboard, the MCP server's detection tools and the HTML, PDF and Markdown reports. It does not condition on the task label, and none of its thresholds decides anything in the paper: use it to explore, and the audit to measure.

### Several methods at once

```python
from shortcut_detect import ShortcutDetector, generate_linear_shortcut

# 500 embeddings of width 64; the first 5 dimensions separate the two groups
X, group = generate_linear_shortcut(n_samples=500, embedding_dim=64, shortcut_dims=5, seed=42)

detector = ShortcutDetector(methods=["hbac", "probe", "statistical", "geometric"], seed=42)
detector.fit(X, group)
print(detector.summary())

results = detector.get_results()
print(results["probe"]["risk_value"], results["probe"]["results"]["metrics"]["metric_value"])

detector.generate_report("shortcut_report.html", format="html")   # also "pdf" and "markdown"
```

`fit(embeddings, labels, group_labels=None)` takes the task label as `labels` and the attribute as `group_labels`. When `group_labels` is omitted, the group-based methods use `labels`, which is what the call above does. `generate_no_shortcut(n_samples=500, embedding_dim=64, seed=42)` is the matching negative control.

### One method at a time

Each method is also a class with `fit`, `shortcut_detected_`, `get_report()` and `summary()`.

HBAC splits the embeddings with k-means a few times, then checks cluster purity and linear separability:

```python
from shortcut_detect import HBACDetector

hbac = HBACDetector(max_iterations=3, min_cluster_size=0.05, random_state=42)
hbac.fit(X, group)

verdict = hbac.shortcut_report_["has_shortcut"]
print(hbac.shortcut_detected_, verdict["confidence"], verdict["types"])
print(f"linear test accuracy {hbac.shortcut_report_['linearity_score']['test_accuracy']:.2f}")
print([round(c["purity"], 2) for c in hbac.shortcut_report_["cluster_purities"]])
```

The probe trains a classifier to predict the group from the embedding and scores it on a held-out split:

```python
from sklearn.linear_model import LogisticRegression
from shortcut_detect import SKLearnProbe

probe = SKLearnProbe(LogisticRegression(max_iter=1000), metric="accuracy",
                     evaluation="holdout", test_size=0.2, random_state=42)
probe.fit(X, group)
print(f"held-out accuracy {probe.metric_value_:.3f}, flagged: {probe.shortcut_detected_}")
```

The statistical test compares the groups dimension by dimension and corrects for multiple testing:

```python
from scipy.stats import mannwhitneyu
from shortcut_detect import GroupDiffTest

test = GroupDiffTest(test=mannwhitneyu)
test.fit(X, group)
corrected = test.apply_correction(alpha=0.05, method="fdr_bh", verbose=False)
comparison, significant = next(iter(corrected["significant_features"].items()))
print(f"{comparison}: {len(significant or [])} of {X.shape[1]} dimensions differ after BH correction")
```

The geometric analyzer measures how far the group centroids sit apart and how much the groups' principal subspaces overlap:

```python
from shortcut_detect import GeometricShortcutAnalyzer

geo = GeometricShortcutAnalyzer(n_components=5)
geo.fit(X, group)
pair = geo.bias_pairs_[0]
print(geo.summary_["risk_level"], f"effect size {pair.effect_size:.2f} for groups {pair.groups}")
print(f"mean subspace cosine {geo.subspace_pairs_[0].mean_cosine:.2f}")
```

### Fixed thresholds

These are the rules each class applies; they are heuristics for exploration.

| Method | Flagged when | Default settings |
|--------|--------------|------------------|
| HBAC | two or more clusters have purity above 0.8, or few dimensions matter and the linear test accuracy is above 0.85 | `max_iterations=3`, `min_cluster_size=0.01` |
| Probe | the held-out metric exceeds `threshold` | `metric="f1"`, `threshold=0.70` |
| Statistical | any dimension stays significant after correction | `apply_correction(alpha=0.05, method="fdr_bh")` |
| Geometric | a centroid pair has effect size at or above `effect_threshold`, or a subspace pair has mean cosine at or above `subspace_cosine_threshold` | `0.8` and `0.85` |

## Embedding-only mode

For systems that only expose embeddings (for example a Hugging Face inference endpoint), pass the raw inputs and an embedding source:

```python
from shortcut_detect import ShortcutDetector, HuggingFaceEmbeddingSource

texts = ["patient has mild edema", "normal study", ...]
labels = np.array([1, 0, ...])
groups = np.array([0, 1, ...])

hf_source = HuggingFaceEmbeddingSource(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    pooling="mean",
    batch_size=32,
)

detector = ShortcutDetector(methods=["probe", "statistical"])
detector.fit(
    embeddings=None,  # triggers embedding-only mode
    labels=labels,
    group_labels=groups,
    raw_inputs=texts,
    embedding_source=hf_source,
    embedding_cache_path="embeddings.npy",  # optional cache
)
```

This needs the `hf` extra. See the [embedding sources API](../api/embedding-sources.md) for `CallableEmbeddingSource` and caching.

## Next steps

- [Interactive dashboard](dashboard.md): visual exploration
- [Detection methods](../methods/overview.md): each method in depth
- [API reference](../api/shortcut-detector.md): full API documentation
- [Examples](../examples/basic.md): notebook examples
