# Reporting API

The reporting module turns a fitted `ShortcutDetector` into HTML, PDF, Markdown and CSV, and holds the plot helpers those reports embed. The audit's own report lives in `shortcut_detect.audit.report`; see the [Audit API](audit.md).

## Module reference

::: shortcut_detect.reporting
    options:
      show_root_heading: true
      members:
        - ReportBuilder
        - export_to_csv
        - generate_all_plots

## ReportBuilder

### Constructor

```python
ReportBuilder(detector: ShortcutDetector)
```

Takes a fitted detector; everything in the report comes from `detector.get_results()`.

### Methods

#### to_html()

```python
def to_html(output_path: str, include_visualizations: bool = True)
```

A self-contained HTML report with one section per method and the overall assessment.

#### to_pdf()

```python
def to_pdf(output_path: str, include_visualizations: bool = True)
```

The same report through WeasyPrint; needs its system libraries (see [Installation](../getting-started/installation.md#pdf-export)).

#### to_markdown()

```python
def to_markdown(output_path: str, include_visualizations: bool = False)
```

#### to_csv()

```python
def to_csv(output_dir: str) -> dict[str, str]
```

### Usage

```python
from shortcut_detect.reporting import ReportBuilder

builder = ReportBuilder(detector)
builder.to_html("report.html")
builder.to_markdown("report.md")
files = builder.to_csv("results/")
```

`detector.generate_report(path, format=...)` wraps the same calls.

---

## export_to_csv

```python
def export_to_csv(detector: ShortcutDetector, output_dir: str) -> dict
```

Writes one CSV per result table and returns `{name: path}`. Which files appear depends on the methods that ran; among them `overall_summary.csv`, `hbac_cluster_purities.csv`, `hbac_dimension_importance.csv`, `probe_results.csv` and `statistical_pvalues.csv`.

```python
from shortcut_detect.reporting import export_to_csv

files = export_to_csv(detector, "./results")
print(files)
```

---

## Visualizations

`shortcut_detect.reporting.visualizations` holds the helpers the HTML report embeds. Each returns a base64-encoded PNG string, ready for an `<img src="data:image/png;base64,...">` tag; `generate_interactive_3d_plot` returns Plotly HTML instead.

```python
from shortcut_detect.reporting.visualizations import (
    generate_pca_plot,
    generate_tsne_plot,
    generate_dimension_importance_plot,
    generate_pvalue_heatmap,
    generate_cluster_purity_plot,
    generate_interactive_3d_plot,
    generate_static_3d_plot,
    generate_all_plots,
)

png = generate_pca_plot(embeddings, group_labels, title="PCA by group")
png = generate_tsne_plot(embeddings, group_labels, perplexity=30)

results = detector.get_results()
png = generate_cluster_purity_plot(results["hbac"]["report"]["cluster_purities"])
png = generate_pvalue_heatmap(results["statistical"]["p_values"], results["statistical"]["corrected_pvalues"])
png = generate_dimension_importance_plot(results["hbac"]["report"]["dimension_importance"], top_k=15)

html = generate_interactive_3d_plot(embeddings, group_labels, method="pca")
plots = generate_all_plots(detector)   # {name: image} for every plot the fitted methods support
```

To save one of the PNGs:

```python
import base64

with open("pca.png", "wb") as fh:
    fh.write(base64.b64decode(png))
```

## Integration with ShortcutDetector

The shortest path to a report:

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=['hbac', 'probe', 'statistical'])
detector.fit(embeddings, task_labels, group_labels=group_labels)

# Direct report generation
detector.generate_report("report.html", format="html")
detector.generate_report("report.pdf", format="pdf")
```

## See Also

- [Quick start](../getting-started/quickstart.md)
- [Dashboard](../getting-started/dashboard.md)
- [ShortcutDetector API](shortcut-detector.md)
