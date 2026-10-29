# Interactive Dashboard

ShortKit-ML includes an interactive Gradio web interface for visual exploration and analysis.

## Launching the Dashboard

```bash
# Ensure dashboard dependencies are installed
pip install "shortkit-ml[dashboard]"

# Launch the dashboard
python app.py
```

The dashboard opens at [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Features

### Sample Data

The default data source is **Sample Data (CheXpert - Lightweight)**:

- 2,000 rows drawn at random (`random_state=42`) from the CheXpert `train.csv`, with real metadata and demographics when Stanford's `train.csv` (and `CHEXPERT DEMO.xlsx`) are in `data/`; without them, 2,000 rows of synthetic metadata (sex, race, a Cardiomegaly label) are generated and the run is labelled "synthetic metadata"
- 512-dimensional **synthetic** embeddings generated over that metadata (the first 10 dimensions carry a planted shortcut)
- Race groups (WHITE, BLACK/AFRICAN AMERICAN, ASIAN, OTHER) from the demographics file
- Tick **Use real CheXpert embeddings** to load `data/chest_embeddings.npy`, `chest_labels.npy` and `chest_group_labels.npy` instead, when they exist

These sample embeddings are not the arrays used in the paper; those are on the gated Hugging Face dataset described in [Reproducing the Paper](../reproducibility.md#revised-paper-cmpb-revision-2026).

### Raw Text Mode

With **Input Mode** set to *Use Raw Data*, upload a CSV with a `text` column and name a HuggingFace model (for example `sentence-transformers/all-MiniLM-L6-v2`); the dashboard embeds the text before running detection. This needs the `hf` extra (`pip install "shortkit-ml[hf]"`), which pins `transformers>=4.39,<5`.

### Custom Data Upload

Upload your own data as a CSV file:

```csv
embedding_0,embedding_1,embedding_2,...,task_label,group_label
0.123,0.456,0.789,...,1,group_a
0.234,0.567,0.890,...,0,group_b
...
```

**Requirements:**

- Columns named `embedding_0`, `embedding_1`, etc. for embedding dimensions
- Column named `task_label` for task labels (optional)
- Column named `group_label` for protected attribute labels
- Optional: `attr_<name>` columns (e.g., `attr_race`, `attr_gender`) for Causal Effect

### Detection Methods

The **Detection** tab offers 16 methods; the first ten are selected by default:

- **HBAC** - Clustering analysis
- **Probe** - Classifier-based detection
- **Statistical** - Hypothesis testing
- **Geometric** - Subspace analysis
- **Equalized Odds** - TPR/FPR gaps across groups
- **GroupDRO** - Worst-group robustness
- **Demographic Parity** - Positive rate gaps across groups
- **Intersectional** - Gaps across intersections of attributes (needs two or more attributes: `attr_*` columns, or `group_label` plus one `attr_*`)
- **Bias Direction PCA** - Projection gap along the bias direction
- **Early-Epoch Clustering (SPARE)** - Clusters of early-epoch representations
- **GCE** - High-loss minority samples
- **SIS** - Sufficient input subsets
- **SSA** - Spread Spurious Attribute; the dashboard builds the labelled/unlabelled split (see *SSA Settings*)
- **Causal Effect** - Spurious attribute detection (requires `attr_*` columns)
- **Frequency** - Signal concentrated in few dimensions
- **Generative CVAE** - Counterfactual embedding shifts

An **Overall Assessment Condition** dropdown picks how method results are aggregated (`indicator_count` by default), and a **Multiple Testing Correction** accordion sets the correction for the statistical test. These risk levels and conditions are the interactive detector's heuristics; the paper's audit (`shortcut_detect.audit.conditional_audit`) uses the conditional probe with a permutation null instead, and none of the thresholds here decide anything in the paper.

**Run Detection** fits the selected methods and renders the results panel. Methods that fail or are skipped are listed in a *method(s) did not run* box with the error, rather than dropped silently. The HTML report includes a section per method, among them SSA and Generative CVAE sections when those methods ran.

### Visualizations

The HTML report embeds PCA and t-SNE projections coloured by group, a 3D projection, dimension-importance and p-value heatmaps from the statistical test, and HBAC cluster purities.

### Model Comparison

The **Model Comparison** tab runs the same methods on two or more embedding sets (several CSVs, or several HuggingFace models over one raw-text CSV) and reports them side by side (**Run Comparison**).

### Advanced Analysis

The **Advanced Analysis** tab holds optional tools that do not need a detection run first. Each accordion has one button:

| Accordion | Button | What it does |
|-----------|--------|--------------|
| GradCAM Attention Overlap | Generate GradCAM Heatmaps | Heatmaps for uploaded images and a torch model |
| GradCAM GT Mask Overlap | Run Mask Overlap | Overlap between heatmaps and ground-truth masks |
| SpRAy Heatmap Clustering | Run SpRAy Clustering | Spectral clustering of heatmaps |
| CAV Concept Testing | Run CAV Analysis | Concept activation vectors from a concept bundle |
| VAE Image Shortcut Detection | Run VAE Analysis | VAE latent disentanglement on images |
| Shortcut Feature Masking (M01) | Run Shortcut Masking | Mask or inpaint image regions, or zero/randomise embedding dimensions |
| Background Randomization (M02) | Run Background Randomization | Swap foregrounds across backgrounds |
| Adversarial Debiasing (M04) | Run Adversarial Debiasing | Gradient-reversal encoder; returns debiased embeddings as CSV |
| Last Layer Retraining (M06 DFR) | Run Last Layer Retraining | Refits an L1 logistic head (C and penalty adjustable) on a subset balanced over attribute x task label groups (2A+Y), as in the paper; returns `dfr_prediction` |
| Contrastive Debiasing (M07 CNC) | Run Contrastive Debiasing | Positives are same task, other group; negatives other task; returns debiased embeddings |
| Explanation Regularization (M05 RRR) | Run Explanation Regularization | Penalises input gradients on shortcut regions; needs a model and images |

The three embedding-level mitigations and DFR can read the embeddings exported by the last detection run (*Use embeddings from last detection run*) or an uploaded CSV with `task_label` and a group column. You can also include CAV in the main Detection report by uploading a concept bundle.

### Report Export

Export your analysis:

- **HTML Report** - Interactive report with all visualizations
- **PDF Report** - Printable summary document
- **CSV Export** - Raw results for further analysis (ZIP file)

## Configuration Options

### Method Parameters

These are the controls exposed in the Detection tab's accordions; HBAC, Probe, Statistical and Geometric run with their package defaults and have no dashboard controls.

| Parameter | Default | Description |
|-----------|---------|-------------|
| Significance level (alpha) | 0.05 | Alpha for the statistical test (0.01 to 0.10) |
| Correction method | `fdr_bh` | Multiple testing correction: `fdr_bh`, `bonferroni`, `holm`, `fdr_by` |
| SSA labeled fraction | 0.2 | Portion of samples treated as labeled for SSA (plus a split seed, 42) |
| Frequency top percent / TPR / FPR | 0.05 / 0.5 / 0.15 | Dimension share kept and per-class flagging thresholds; probe evaluation on `train` |
| Early-epoch clustering min cluster ratio | 0.10 | SPARE minimum cluster ratio |
| GCE q / loss percentile / max iterations | 0.7 / 90 / 500 | Generalized cross-entropy settings |
| Causal effect spurious threshold | 0.1 | Attributes with \|effect\| below this are flagged |
| CAV concept quality threshold | 0.7 | Minimum concept AUC (Detection tab bundle and Advanced Analysis) |
| VAE latent dimension | 10 | Advanced Analysis |

Projections in the report (PCA, t-SNE, 3D) use fixed settings; there are no visualization controls in the dashboard.

## Screenshot

![Dashboard Interface](../assets/dashboard-preview.png)

## Troubleshooting

### Dashboard Won't Start

```bash
# Check Gradio is installed
python -c "import gradio; print(gradio.__version__)"

# Reinstall if needed
pip install --upgrade gradio
```

### Port Already in Use

The app picks the first free port from 7860 upwards. To force one:

```bash
GRADIO_SERVER_PORT=7861 python app.py
```

## Programmatic Access

`app.py` builds a `gr.Blocks` object named `demo` at import time and launches it only under `if __name__ == "__main__":`

```python
from app import demo

demo.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,
    share=True,  # Create public link
)
```

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[dashboard]"

EXPOSE 7860
CMD ["python", "app.py"]
```

```bash
docker build -t shortcut-detect-dashboard .
docker run -p 7860:7860 shortcut-detect-dashboard
```

## Next Steps

- [Quick Start](quickstart.md) - Programmatic usage
- [Detection Methods](../methods/overview.md) - Understand each method
- [Examples](../examples/basic.md) - Jupyter notebooks
