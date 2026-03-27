# Reporting API

The reporting module generates HTML and PDF reports from detection results.

## Module Reference

::: shortcut_detect.reporting
    options:
      show_root_heading: true
      members:
        - ReportBuilder
        - CSVExporter

## ReportBuilder

### Constructor

```python
ReportBuilder(
    title: str = "Shortcut Detection Report",
    include_visualizations: bool = True,
    theme: str = 'default'
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | str | "Shortcut Detection Report" | Report title |
| `include_visualizations` | bool | True | Include plots |
| `theme` | str | 'default' | Report theme |

### Methods

#### add_section()

```python
def add_section(
    title: str,
    content: str,
    visualizations: list = None
) -> ReportBuilder
```

Add a section to the report.

#### add_summary()

```python
def add_summary(detector: ShortcutDetector) -> ReportBuilder
```

Add summary from ShortcutDetector results.

#### build_html()

```python
def build_html(output_path: str) -> None
```

Generate HTML report.

#### build_pdf()

```python
def build_pdf(output_path: str) -> None
```

Generate PDF report (requires WeasyPrint).

### Usage

```python
from shortcut_detect.reporting import ReportBuilder

# Create builder
builder = ReportBuilder(title="My Analysis Report")

# Add detector results
builder.add_summary(detector)

# Add custom sections
builder.add_section(
    title="Custom Analysis",
    content="Additional findings...",
    visualizations=[fig1, fig2]
)

# Generate reports
builder.build_html("report.html")
builder.build_pdf("report.pdf")
```

---

## CSVExporter

Export detection results to CSV files.

### Constructor

```python
CSVExporter(output_dir: str = "exports")
```

### Methods

#### export_all()

```python
def export_all(detector: ShortcutDetector) -> list[str]
```

Export all results to CSV files.

**Returns:** List of created file paths.

#### export_statistical()

```python
def export_statistical(results: dict, path: str) -> str
```

Export statistical test results.

#### export_probe()

```python
def export_probe(results: dict, path: str) -> str
```

Export probe results.

### Usage

```python
from shortcut_detect.reporting import CSVExporter

exporter = CSVExporter(output_dir="./results")

# Export all results
files = exporter.export_all(detector)
print(f"Created: {files}")

# Export specific results
exporter.export_statistical(
    detector.statistical_results_,
    "statistical_results.csv"
)
```

### Output Files

| File | Contents |
|------|----------|
| `summary.csv` | Overall risk assessment |
| `hbac_results.csv` | HBAC metrics |
| `probe_results.csv` | Probe accuracy, predictions |
| `statistical_results.csv` | P-values, effect sizes |
| `geometric_results.csv` | Bias direction, overlap |
| `gradcam_mask_overlap_samples.csv` | Per-sample GradCAM vs. GT mask overlap metrics |
| `cav_concept_scores.csv` | Per-concept CAV quality/TCAV scores and flags |

---

## Visualizations

### Built-in Plots

The reporting module includes visualization helpers:

```python
from shortcut_detect.reporting.visualizations import (
    plot_embedding_scatter,
    plot_volcano,
    plot_feature_importance,
    plot_group_distributions
)

# t-SNE/UMAP scatter plot
fig = plot_embedding_scatter(
    embeddings,
    group_labels,
    method='tsne'
)
fig.savefig("scatter.png")

# Volcano plot
fig = plot_volcano(
    pvalues=test.pvalues_corrected_,
    effect_sizes=test.effect_sizes_
)
fig.savefig("volcano.png")
```

### Interactive Plots

For interactive visualizations:

```python
from shortcut_detect.reporting.visualizations import (
    plot_3d_embedding,
    create_interactive_report
)

# 3D plot with Plotly
fig = plot_3d_embedding(embeddings, group_labels)
fig.write_html("3d_plot.html")

# Full interactive report
create_interactive_report(detector, "interactive_report.html")
```

## Integration with ShortcutDetector

The simplest way to generate reports:

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=['hbac', 'probe', 'statistical'])
detector.fit(embeddings, group_labels)

# Direct report generation
detector.generate_report("report.html", format="html")
detector.generate_report("report.pdf", format="pdf")
```

## See Also

- [Quick Start](../getting-started/quickstart.md)
- [Dashboard](../getting-started/dashboard.md)
- [ShortcutDetector API](shortcut-detector.md)
