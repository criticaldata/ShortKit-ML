# ShortcutDetector API

The `ShortcutDetector` class provides a unified interface for running all shortcut detection methods.

## Class Reference

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

## Quick Reference

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
| `**kwargs` | dict | - | Additional detector-specific configuration |

### Available Methods

| Method Key | Description |
|------------|-------------|
| `'hbac'` | Hierarchical Bias-Aware Clustering |
| `'probe'` | Probe-based classifier detection |
| `'statistical'` | Feature-wise statistical testing |
| `'geometric'` | Geometric subspace analysis |
| `'cav'` | Concept Activation Vector testing (loader mode) |

## Methods

### fit()

```python
def fit(
    embeddings: np.ndarray,
    group_labels: np.ndarray,
    task_labels: np.ndarray = None,
    raw_inputs: list = None,
    embedding_source: EmbeddingSource = None,
    embedding_cache_path: str = None
) -> ShortcutDetector
```

Fit all detection methods on the data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | ndarray | Shape (n_samples, n_features). Pass None for embedding-only mode. |
| `group_labels` | ndarray | Protected attribute labels |
| `task_labels` | ndarray | Optional task labels |
| `raw_inputs` | list | Raw inputs for embedding generation |
| `embedding_source` | EmbeddingSource | Embedding generator for embedding-only mode |
| `embedding_cache_path` | str | Path to cache generated embeddings |

**Returns:** self

### summary()

```python
def summary() -> str
```

Get a human-readable summary of all detection results.

**Returns:** Multi-line string with risk assessment and key metrics.

### generate_report()

```python
def generate_report(
    output_path: str,
    format: str = 'html',
    include_visualizations: bool = True
) -> None
```

Generate a comprehensive report.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | str | required | Path for output file |
| `format` | str | `'html'` | Output format: 'html' or 'pdf' |
| `include_visualizations` | bool | True | Include plots in report |

### get_results()

```python
def get_results() -> dict
```

Get raw results from all methods.

**Returns:** Dictionary with results from each method.

## Attributes (after fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `results_` | dict | Results from all detection methods |
| `detectors_` | dict | Fitted detector instances for each method |
| `embeddings_` | ndarray | Stored embeddings used for fitting |
| `labels_` | ndarray | Stored labels used for fitting |
| `group_labels_` | ndarray | Stored group labels (if provided) |

## Usage Examples

### Basic Usage

```python
from shortcut_detect import ShortcutDetector
import numpy as np

# Load data
embeddings = np.load("embeddings.npy")
group_labels = np.load("groups.npy")

# Create detector with all methods
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical', 'geometric']
)

# Fit
detector.fit(embeddings, group_labels)

# Results
print(detector.summary())
results = detector.get_results()
print(f"Probe accuracy: {results['probe']['accuracy']}")
```

### Custom Parameters

```python
detector = ShortcutDetector(
    methods=['hbac', 'probe', 'statistical'],
    seed=42,
    n_bootstraps=1000,
    probe_backend='sklearn'
)
```

### Custom Overall Assessment

The `condition_name` parameter selects how method-level results are aggregated into
the final risk summary. Five built-in conditions are available:

| Condition | Description | Key Parameters |
|-----------|-------------|----------------|
| `indicator_count` | Counts total risk indicators across methods (default) | - |
| `majority_vote` | Counts methods with at least one indicator as votes | `high_threshold` (int) |
| `weighted_risk` | Weights each detector by evidence strength (probe accuracy, stat significance ratio, HBAC confidence, geometric effect size) | `high_threshold` (float), `moderate_threshold` (float) |
| `multi_attribute` | Cross-references risk across sensitive attributes; escalates when multiple attributes independently flag shortcuts | `high_threshold` (int) |
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

### Embedding-Only Mode

```python
from shortcut_detect import ShortcutDetector, HuggingFaceEmbeddingSource

# Create embedding source
hf_source = HuggingFaceEmbeddingSource(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Detect shortcuts from raw inputs
detector = ShortcutDetector(methods=['probe', 'statistical'])
detector.fit(
    embeddings=None,  # Triggers embedding-only mode
    group_labels=groups,
    raw_inputs=texts,
    embedding_source=hf_source,
    embedding_cache_path="embeddings.npy"
)
```

### Report Generation

```python
# HTML report (recommended)
detector.generate_report(
    output_path="report.html",
    format="html",
    include_visualizations=True
)

# PDF report
detector.generate_report(
    output_path="report.pdf",
    format="pdf"
)
```

### Accessing Raw Results

```python
# Get all results
results = detector.get_results()

# Access specific method results
hbac = results['hbac']
print(f"HBAC purity: {hbac['purity']:.2f}")

probe = results['probe']
print(f"Probe accuracy: {probe['accuracy']:.2%}")

statistical = results['statistical']
print(f"Significant features: {statistical['n_significant']}")

geometric = results['geometric']
print(f"Effect size: {geometric['effect_size']:.2f}")
```

### Loader-based CAV Usage

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

## Error Handling

```python
try:
    detector.fit(embeddings, group_labels)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Detection failed: {e}")
```

## See Also

- [Quick Start Guide](../getting-started/quickstart.md)
- [Detection Methods Overview](../methods/overview.md)
- [HBAC API](hbac.md)
- [Probes API](probes.md)
- [Statistical API](statistical.md)
- [Geometric API](geometric.md)
