# ShortcutDetector API

`ShortcutDetector` runs any of the 19 registered methods on one embedding matrix, compares each against its fixed threshold and aggregates the flags into a risk label. It is the interactive pipeline behind the dashboard and the reports; the paper's measurement is [`conditional_audit`](audit.md).

## Class reference

::: shortcut_detect.unified.ShortcutDetector
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - fit
        - summary
        - generate_report
        - get_results

## Quick reference

### Constructor

```python
ShortcutDetector(
    methods: list[str] = ['hbac', 'probe', 'statistical'],
    seed: int = 42,
    condition_name: str = 'indicator_count',
    condition_kwargs: dict | None = None,
    **kwargs
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `methods` | list[str] | `['hbac', 'probe', 'statistical']` | Methods to run |
| `seed` | int | `42` | Random seed for reproducibility |
| `condition_name` | str | `'indicator_count'` | Overall assessment condition to use |
| `condition_kwargs` | dict | `None` | Keyword args passed to the selected condition |
| `**kwargs` | dict | - | Settings read by the method builders (see [Custom parameters](#custom-parameters)) |

### Available methods

`shortcut_detect.discovery.available_method_plugins()` returns the 19 keys:

| Method key | Description |
|------------|-------------|
| `'hbac'` | Hierarchical bias-aware clustering |
| `'probe'` | Probe classifier (`probe_backend="sklearn"` or `"torch"`) |
| `'statistical'` | Feature-wise two-sample tests with BH correction |
| `'geometric'` | Centroid distance and subspace overlap |
| `'bias_direction_pca'` | Projection gap along the bias direction |
| `'frequency'` | Signal concentrated in few dimensions |
| `'equalized_odds'`, `'demographic_parity'`, `'intersectional'` | Fairness gaps of a classifier trained on the embeddings |
| `'groupdro'`, `'gce'`, `'ssa'`, `'early_epoch_clustering'` | Training-dynamics methods |
| `'causal_effect'`, `'generative_cvae'`, `'vae'` | Causal and generative methods |
| `'sis'` | Sufficient input subsets |
| `'cav'`, `'gradcam_mask_overlap'` | Loader mode: `fit_from_loaders` with concept sets or heatmaps |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray | None,
    labels: np.ndarray,
    group_labels: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    raw_inputs: Sequence | None = None,
    embedding_source: EmbeddingSource | None = None,
    embedding_cache_path: str | None = None,
    force_embedding_recompute: bool = False,
    splits: dict[str, np.ndarray] | None = None,
    extra_labels: dict[str, np.ndarray] | None = None,
) -> ShortcutDetector
```

Fits every selected method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray or None | Shape (n_samples, embedding_dim); None for embedding-only mode |
| `labels` | ndarray | Task labels, shape (n_samples,) |
| `group_labels` | ndarray or None | Protected attribute, shape (n_samples,). When None, `labels` is used for the group-based methods |
| `feature_names` | list | Optional names for the dimensions |
| `raw_inputs` | sequence | Raw inputs (texts, ids) for `embedding_source` |
| `embedding_source` | EmbeddingSource | Generates embeddings when `embeddings` is None |
| `embedding_cache_path` | str | Cache file for generated embeddings |
| `force_embedding_recompute` | bool | Ignore the cache |
| `splits` | dict | `train_l` and `train_u` index sets for `ssa` |
| `extra_labels` | dict | Extra per-sample arrays (for example several attributes for `intersectional` and `causal_effect`); -1 marks unknown |

**Returns:** self

### summary()

```python
def summary() -> str
```

One block per method (risk label, reason, key numbers) followed by the overall assessment.

### generate_report()

```python
def generate_report(
    output_path: str = None,
    format: str = 'html',
    include_visualizations: bool = True,
    export_csv: bool = False,
    csv_dir: str = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | str | required | Path for the report |
| `format` | str | `'html'` | `'html'`, `'pdf'` or `'markdown'` |
| `include_visualizations` | bool | True | Embed plots |
| `export_csv` | bool | False | Also write the CSV tables |
| `csv_dir` | str | None | Where the CSVs go (default: `csv_results/` next to the report) |

### get_results()

```python
def get_results() -> dict
```

`{method: result dict}`; see [Accessing raw results](#accessing-raw-results).

### fit_from_loaders()

Loader mode for `cav` and `gradcam_mask_overlap`; see [Loader-based CAV usage](#loader-based-cav-usage).

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `results_` | dict | Results from all detection methods |
| `detectors_` | dict | Fitted detector instance per method |
| `embeddings_` | ndarray | Embeddings used for fitting |
| `labels_` | ndarray | Task labels used for fitting |
| `group_labels_` | ndarray | Group labels (`labels` when none were given) |
| `protected_labels_` | ndarray or None | The `group_labels` as passed |

## Usage examples

### Basic usage

```python
import numpy as np
from shortcut_detect import ShortcutDetector

embeddings = np.load("embeddings.npy")
task_labels = np.load("task.npy")
group_labels = np.load("groups.npy")

detector = ShortcutDetector(methods=['hbac', 'probe', 'statistical', 'geometric'], seed=42)
detector.fit(embeddings, task_labels, group_labels=group_labels)

print(detector.summary())
results = detector.get_results()
print(f"Probe score: {results['probe']['results']['metrics']['metric_value']:.2f} ({results['probe']['risk_value']})")
```

### Custom parameters

Keyword arguments reach the method builders; each reads the ones it knows:

```python
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical'],
    seed=42,
    probe_backend='sklearn',   # or 'torch'
    hbac_target='attribute',   # or 'task'
)
```

Other builder settings include `dp_gap_threshold`, `eo_tpr_gap_threshold`, `eo_fpr_gap_threshold`, `freq_target`, `device` and `epochs`; see each builder in `shortcut_detect/<method>/builder.py`.

### Custom overall assessment

The `condition_name` parameter selects how method-level results are aggregated into
the final risk summary. Five built-in conditions are available:

| Condition | Description | Key parameters |
|-----------|-------------|----------------|
| `indicator_count` | Counts total risk indicators across methods (default) | - |
| `majority_vote` | Counts methods with at least one indicator as votes | `high_threshold` (int) |
| `weighted_risk` | Weights each detector by evidence strength (probe score, stat significance ratio, HBAC confidence, geometric effect size) | `high_threshold` (float), `moderate_threshold` (float) |
| `multi_attribute` | Cross-references risk across sensitive attributes; escalates when several attributes flag shortcuts on their own | `high_threshold` (int) |
| `meta_classifier` | Trained sklearn meta-classifier on detector features, or heuristic fallback | `model_path` (str), `high_threshold` (float), `moderate_threshold` (float) |

```python
# Majority vote
detector = ShortcutDetector(
    methods=['probe', 'statistical'],
    condition_name='majority_vote',
    condition_kwargs={'high_threshold': 2},
)
```

```python
# Weighted risk scoring
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric'],
    condition_name='weighted_risk',
    condition_kwargs={'high_threshold': 0.6, 'moderate_threshold': 0.3},
)
```

```python
# Multi-attribute intersection
detector = ShortcutDetector(
    methods=['probe', 'statistical'],
    condition_name='multi_attribute',
    condition_kwargs={'high_threshold': 2},
)
```

```python
# Meta-classifier (heuristic fallback when no trained model provided)
detector = ShortcutDetector(
    methods=['probe', 'statistical', 'hbac'],
    condition_name='meta_classifier',
)

# Meta-classifier with a trained model
detector = ShortcutDetector(
    methods=['probe', 'statistical', 'hbac'],
    condition_name='meta_classifier',
    condition_kwargs={'model_path': 'path/to/meta_model.joblib'},
)
```

The `meta_classifier` condition also exposes `MetaClassifierCondition.extract_features(ctx)`
for building training datasets from synthetic benchmark runs.

The default `indicator_count` condition preserves the library's existing summary semantics.

### Embedding-only mode

```python
from shortcut_detect import ShortcutDetector, HuggingFaceEmbeddingSource

hf_source = HuggingFaceEmbeddingSource(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

detector = ShortcutDetector(methods=['probe', 'statistical'])
detector.fit(
    embeddings=None,          # triggers embedding-only mode
    labels=task_labels,
    group_labels=groups,
    raw_inputs=texts,
    embedding_source=hf_source,
    embedding_cache_path="embeddings.npy",
)
```

### Report generation

```python
detector.generate_report(output_path="report.html", format="html", include_visualizations=True)
detector.generate_report(output_path="report.pdf", format="pdf")           # needs WeasyPrint's system libraries
detector.generate_report(output_path="report.md", format="markdown")
detector.generate_report(output_path="report.html", format="html", export_csv=True, csv_dir="csv_results")
```

### Accessing raw results

```python
results = detector.get_results()

hbac = results['hbac']
print(hbac['risk_value'], hbac['report']['has_shortcut'])

probe = results['probe']
print(probe['risk_value'], probe['results']['metrics'])

statistical = results['statistical']
print(statistical['risk_value'], {k: len(v or []) for k, v in statistical['significant_features'].items()})

geometric = results['geometric']
print(geometric['risk_value'], geometric['summary'], geometric['bias_pairs'][0].effect_size)
```

Every entry carries `detector` (the fitted instance), `summary_title`, `summary_lines`, `risk_indicators`, `success`, `risk_value`, `risk_label` and `risk_reason`; the method-specific keys above sit next to them. A method that failed has `success=False` and an `error`.

### Loader-based CAV usage

```python
detector = ShortcutDetector(methods=["cav"])
detector.fit_from_loaders({
    "cav": {
        "concept_sets": {"shortcut": concept_arr},
        "random_set": random_arr,
        "target_directional_derivatives": directional_derivatives,  # optional but needed for TCAV risk
    }
})
print(detector.get_results()["cav"]["metrics"])
```

## Error handling

```python
try:
    detector.fit(embeddings, task_labels, group_labels=group_labels)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Detection failed: {e}")
```

## See also

- [Quick start](../getting-started/quickstart.md)
- [Detection methods overview](../methods/overview.md)
- [Audit API](audit.md)
- [HBAC API](hbac.md)
- [Probes API](probes.md)
- [Statistical API](statistical.md)
- [Geometric API](geometric.md)
