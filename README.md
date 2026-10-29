<p align="center">
  <img src="docs/assets/shortkit.png" alt="ShortKit-ML Logo" width="400"/>
</p>

> **ShortKit-ML** detects and mitigates shortcuts and biases in machine learning embedding spaces. 20+ detection and mitigation methods with a unified API. **Multi-attribute support** tests multiple sensitive attributes simultaneously. Model Comparison mode for benchmarking multiple embedding models.

[![PyPI version](https://img.shields.io/pypi/v/shortkit-ml.svg)](https://pypi.org/project/shortkit-ml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![CI](https://github.com/criticaldata/ShortKit-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/criticaldata/ShortKit-ML/actions/workflows/tests.yml)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-ShortKIT--ML--data-yellow.svg)](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data)
[![Docs](https://img.shields.io/badge/docs-criticaldata.github.io-blue.svg)](https://criticaldata.github.io/ShortKit-ML/)
[![Paper](https://img.shields.io/badge/paper-medRxiv-red.svg)](https://www.medrxiv.org/content/10.64898/2026.04.29.26352053v1)


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detection Methods](#detection-methods)
- [Overall Assessment Conditions](#overall-assessment-conditions)
- [MCP Server](#mcp-server)
- [Reproducing Paper Results](#reproducing-paper-results)
- [GPU Support](#gpu-support)
- [Interactive Dashboard](#interactive-dashboard)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

ShortKit-ML provides a comprehensive toolkit for detecting and mitigating shortcuts (unwanted biases) in embedding spaces:

- **20+ detection methods**: HBAC, Probe, Statistical, Geometric, Bias Direction PCA, Equalized Odds, Demographic Parity, Intersectional, GroupDRO, GCE, Causal Effect, SSA, SIS, CAV, VAE, Early-Epoch Clustering, and more
- **6 mitigation methods**: Shortcut Masking, Background Randomization, Adversarial Debiasing, Explanation Regularization, Last Layer Retraining, Contrastive Debiasing
- **5 pluggable risk conditions**: indicator_count, majority_vote, weighted_risk, multi_attribute, meta_classifier

**Key Features:**
- Unified `ShortcutDetector` API for all methods
- Interactive Gradio dashboard with real-time analysis
- PDF/HTML/Markdown reports with visualizations
- Embedding-only mode (no model access needed)
- Multi-attribute support: test race, gender, age simultaneously
- Model Comparison mode: compare multiple embedding models side-by-side

## Installation

Available on PyPI at **[pypi.org/project/shortkit-ml](https://pypi.org/project/shortkit-ml/)**.

```bash
pip install shortkit-ml
```

For all optional extras (dashboard, reporting, VAE, HuggingFace, MCP, etc.):

```bash
pip install "shortkit-ml[all]"
```

The PyPI release 0.1.0 predates the `shortcut_detect.audit` module used by the revised paper; install from source (`pip install -e .`) until the next release.

### Development Install (from source)

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd ShortKit-ML
pip install -e ".[all]"
```

Or with `uv`:

```bash
uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[all]"
```

### Optional: PDF Export Dependencies

```bash
# macOS
brew install pango gdk-pixbuf libffi
# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0
```

> HTML and Markdown reports work without these. PDF export is optional.

## Quick Start

```python
from shortcut_detect import ShortcutDetector
import numpy as np

embeddings = np.load("embeddings.npy")  # (n_samples, embedding_dim)
labels = np.load("labels.npy")          # (n_samples,)

detector = ShortcutDetector(methods=['hbac', 'probe', 'statistical', 'geometric', 'equalized_odds'])
detector.fit(embeddings, labels)

detector.generate_report("report.html", format="html")
print(detector.summary())
```

### Embedding-Only Mode

For closed-source models or systems that only expose embeddings:

```python
from shortcut_detect import ShortcutDetector, HuggingFaceEmbeddingSource

hf_source = HuggingFaceEmbeddingSource(model_name="sentence-transformers/all-MiniLM-L6-v2")
detector = ShortcutDetector(methods=["probe", "statistical"])
detector.fit(embeddings=None, labels=labels, group_labels=groups,
             raw_inputs=texts, embedding_source=hf_source)
```

> See [Embedding-Only Guide](docs/api/embedding-sources.md) for `CallableEmbeddingSource` and caching options.

## Detection Methods

| Method | Key | What It Detects | Reference |
|--------|-----|-----------------|-----------|
| **HBAC** | `hbac` | Clustering by protected attributes | - |
| **Probe** | `probe` | Group info recoverable from embeddings | - |
| **Statistical** | `statistical` | Dimensions with group differences | - |
| **Geometric** | `geometric` | Bias directions & prototype overlap | - |
| **Bias Direction PCA** | `bias_direction_pca` | Projection gap along bias direction | Bolukbasi 2016 |
| **Equalized Odds** | `equalized_odds` | TPR/FPR disparities | Hardt 2016 |
| **Demographic Parity** | `demographic_parity` | Prediction rate disparities | Feldman 2015 |
| **Early Epoch Clustering** | `early_epoch_clustering` | Shortcut reliance in early reps | Yang 2023 |
| **GCE** | `gce` | High-loss minority samples | - |
| **Frequency** | `frequency` | Signal in few dimensions | - |
| **GradCAM Mask Overlap** | `gradcam_mask_overlap` | Attention overlap with shortcut masks | - |
| **SpRAy** | `spray` | Spectral clustering of heatmaps | Lapuschkin 2019 |
| **CAV** | `cav` | Concept-level sensitivity | Kim 2018 |
| **Causal Effect** | `causal_effect` | Spurious attribute influence | - |
| **VAE** | `vae` | Latent disentanglement signatures | - |
| **SSA** | `ssa` | Semi-supervised spectral shift | [arXiv:2204.02070](https://arxiv.org/abs/2204.02070) |
| **Generative CVAE** | `generative_cvae` | Counterfactual embedding shifts | - |
| **GroupDRO** | `groupdro` | Worst-group performance gaps | Sagawa 2020 |
| **SIS** | `sis` | Sufficient input subsets (minimal dims for prediction) | Carter 2019 |
| **Intersectional** | `intersectional` | Intersectional fairness gaps (2+ attributes) | Buolamwini 2018 |

### Mitigation Methods

| Method | Class | Strategy | Reference |
|--------|-------|----------|-----------|
| **Shortcut Masking** | `ShortcutMasker` | Zero/randomize/inpaint shortcut regions | - |
| **Background Randomization** | `BackgroundRandomizer` | Swap foreground across backgrounds | - |
| **Adversarial Debiasing** | `AdversarialDebiasing` | Remove group information adversarially | Zhang 2018 |
| **Explanation Regularization** | `ExplanationRegularization` | Penalize attention on shortcuts (RRR) | Ross 2017 |
| **Last Layer Retraining** | `LastLayerRetraining` | Retrain final layer balanced (DFR) | Kirichenko 2023 |
| **Contrastive Debiasing** | `ContrastiveDebiasing` | Contrastive loss to align groups (CNC) | - |

> See [Detection Methods Overview](docs/methods/overview.md) for per-method usage, interpretation guides, and code examples.

## Overall Assessment Conditions

`ShortcutDetector` supports pluggable risk aggregation conditions that control how method-level results map to the final HIGH/MODERATE/LOW summary.

| Condition | Best For | Description |
|-----------|----------|-------------|
| `indicator_count` | General use (default) | Count of risk signals: 2+ = HIGH, 1 = MODERATE, 0 = LOW |
| `majority_vote` | Conservative screening | Consensus across methods |
| `weighted_risk` | Nuanced analysis | Evidence strength matters (probe accuracy, effect sizes, etc.) |
| `multi_attribute` | Multi-demographic | Escalates when multiple attributes flag risk |
| `meta_classifier` | Trained pipelines | Logistic regression meta-model on detector outputs (bundled model included) |

```python
detector = ShortcutDetector(
    methods=["probe", "statistical"],
    condition_name="weighted_risk",
    condition_kwargs={"high_threshold": 0.6, "moderate_threshold": 0.3},
)
```

Custom conditions can be registered via `@register_condition("name")`. See [Conditions API](docs/api/shortcut-detector.md) for details.

## MCP Server

ShortKit-ML ships an [MCP](https://modelcontextprotocol.io/) server so AI assistants (Claude, Cursor, GitHub Copilot in VS Code, etc.) can call detection tools directly from chat, with no Python script required.

```bash
uv pip install -e ".[mcp]"
```

### GitHub Copilot in VS Code

1. After installing the MCP extra, open VS Code and go to Settings (Ctrl+,).
2. Search for "GitHub Copilot Chat: MCP" and open the JSON settings.
3. Copy the server configuration from the repo's `.vscode/mcp.json` file into the `github.copilot.chat.mcp` object in your settings.
4. Ensure the `"command"` path points to your virtual environment's Python (e.g., update `.venv/bin/python` to the full path if needed).
5. Restart VS Code or reload the GitHub Copilot Chat extension.
6. The ShortKit-ML tools will now be available in Copilot Chat sessions.

Full setup instructions for **Claude Code**, **Claude Desktop**, and **GitHub Copilot in VS Code**, available tools, file-based input, and troubleshooting: [**criticaldata.github.io/ShortKit-ML/mcp-server**](https://criticaldata.github.io/ShortKit-ML/mcp-server/).

## Reproducing Paper Results

### Revised paper (CMPB revision)

This is the supported path. Every table, figure and number of the revised paper (under revision at *Computer Methods and Programs in Biomedicine*) is generated by one pipeline, which reads the gated hub data under `data/` (download it as described under [Data](#data); `train.csv` and `CHEXPERT DEMO.xlsx` come from Stanford, see the "Files not on the hub" table in `data/README.md`):

```bash
./scripts/reproduce_paper.sh revision   # same as scripts/revision/run_all.sh
```

The paper's audit is `shortcut_detect.audit.conditional_audit`. One primary test, the conditional probe (L2 logistic regression predicting the attribute from the embedding and the task label, out-of-fold AUROC), is calibrated by permuting the attribute within task-label strata, at patient level where patient ids are given, with K=500 permutations in the clinical audits. Five secondary detectors (statistical, geometric, bias direction PCA, HBAC, frequency) share the same null with K=200 permutations and are reported as diagnostics; the number of flagged detectors in a cell is descriptive only. SIS, the fairness gaps and the training-dynamics methods are outside the score. The audit reports shortcut potential: a positive result says the attribute is recoverable from the embedding given the task. Whether a classifier uses it is a separate question, tested in the head experiment. Results go to `~/Documents/shortkit-ml-results/2026-10-28_revision_<commit>/` unless `SHORTKIT_REVISION_DIR` is set; the `tables` stage writes to `output/paper_tables/revision/`. See the [reproducibility page](docs/reproducibility.md) for the stages and output folders.

### First submission (archived)

The commands below belong to the first submission (13 methods compared against fixed thresholds) and are kept for the record so that its tables can be replayed. They do not produce the revised paper's results. All rows assume `pip install -e ".[all]"`. The synthetic rows run on a fresh clone; the others need the files in the "Needs" column, and the rows are in dependency order.

| Step | Command | Needs | Output |
|------|---------|-------|--------|
| Synthetic grid | `python -m shortcut_detect.benchmark.paper_run --config examples/paper_benchmark_config_reproducible.json` | nothing; edit the config to change effect sizes, sample sizes, imbalance ratios and embedding dimensionalities | `output/paper_benchmark/` (CSVs, figures, summary markdown) |
| Synthetic profiles | `./scripts/reproduce_paper.sh smoke` (also `default` and `full`; `--help` lists them) | nothing | `output/paper_benchmark_<profile>_<timestamp>/` |
| Paper figures | `python scripts/generate_paper_figures.py` | `data/chest_embeddings.npy`, `chest_labels.npy`, `chest_group_labels.npy` (hub root, first-submission files) | `output/paper_figures/` |
| CheXpert MedCLIP artefacts | `python scripts/run_dataset2_benchmark.py --manifest data/chexpert_manifest.csv --model medclip --root . --artifacts-dir output/paper_benchmark/chexpert_embeddings --config examples/paper_benchmark_config_reproducible.json` | the archived pickle `data/chexpert/medclip/embedding_from_medclip.pkl` (not on the hub; `biomedclip` and `cxr-foundation` need their own pickles) plus `data/chexpert_manifest.csv` (hub root). The hub's `data/chexpert/medclip_embeddings.npy` and `medclip_manifest.csv` are this step's output: copy them to `output/paper_benchmark/chexpert_embeddings/medclip_embeddings.npy` and `medclip_metadata.csv` to skip it | `output/paper_benchmark/chexpert_embeddings/` |
| CheXpert benchmark | `python scripts/run_chexpert_benchmark.py` | `output/paper_benchmark/chexpert_embeddings/` from the previous row | `output/paper_benchmark/chexpert_results/` |
| MIMIC-CXR benchmark | `python scripts/run_mimic_benchmark.py` | `data/mimic_cxr/` (hub). `scripts/setup_mimic_cxr_data.py` rebuilds those files from the coreset parquet files (`mimic_cxr/source/` on the hub) and is not needed otherwise. Took 8 minutes on a laptop | `output/paper_benchmark/mimic_cxr_results/` |
| CelebA benchmark | `python scripts/setup_celeba_data.py` then `python scripts/run_celeba_benchmark.py` | the setup script writes `data/celeba/celeba_embeddings.npy` and its CSVs (downloaded attributes, or synthetic data when the download fails); these are not on the hub | `output/paper_benchmark/celeba_results/aggregate_report.json` |
| CelebA real benchmark | `python scripts/run_celeba_real_benchmark.py` | `data/celeba/celeba_real_*` (hub). `scripts/extract_celeba_embeddings.py` regenerates them from the CelebA images. One of the three tasks took over 20 minutes on a laptop | `output/paper_benchmark/celeba_real_results/` |
| First-submission tables | `python scripts/generate_all_paper_tables.py` | `chexpert_results/`, `mimic_cxr_results/` and `celeba_results/aggregate_report.json` from the rows above (tables a to c are synthetic and need nothing) | `output/paper_tables/*.tex` |

### Docker (fully self-contained)

The image runs the synthetic profiles only (`smoke`, `default`, `full`).

```bash
docker build -t shortcut-detect .
docker run --rm -v $(pwd)/output:/app/output shortcut-detect full
```

### Data

> **Important:** The `data/` folder in this repository is empty. All embeddings and metadata are hosted on HuggingFace and must be downloaded separately. Raw CheXpert and MIMIC-CXR images are **not redistributed**.

All embeddings are hosted on HuggingFace (gated, PhysioNet-restricted access). The embeddings, metadata and MIMIC-CXR label file there are derived from credentialed data and are shared only with users who hold the matching access (PhysioNet for MIMIC-CXR, Stanford for CheXpert); see [Requesting access](#requesting-access) below.

**[MITCriticalData/ShortKit-ML-data](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data)**

```bash
# Download all embeddings into data/
hf download MITCriticalData/ShortKit-ML-data --repo-type dataset --local-dir data/
```

| Dataset | Location | Embedding Models | Dim | Samples |
|---------|----------|-----------------|-----|---------|
| Synthetic | Generated at runtime | `SyntheticGenerator(seed=42)` | 128 | Configurable |
| CheXpert (current) | `data/chexpert_multibackbone/` and `data/chexpert_multibackbone/public/` | ResNet-50, DenseNet-121, ViT-B/16, ViT-B/32, DINOv2, RAD-DINO, MedSigLIP-448 | 768-2048 | 2,000 each |
| CheXpert (first submission) | `data/chexpert/`, and the ViT-B/32, DINOv2, RAD-DINO and MedSigLIP arrays in `chexpert_multibackbone/` | MedCLIP and the same seven names | 512-2048 | 2,000 each |
| MIMIC-CXR | `data/mimic_cxr/` | RAD-DINO, ViT-B/16, ViT-B/32, MedSigLIP-448 | 768-1152 | 1,491 each |
| CelebA | `data/celeba/` | ResNet-50 (ImageNet) | 2,048 | 10,000 |

**CheXpert sample.** The 2,000 images are the first 2,000 of the official `train.csv` order (not a random sample), from 477 patients; 423 are lateral views. Sex comes from `train.csv` and race per patient from the CheXpert demographics file, compared as White against non-White in the primary analysis. Cross-validation folds and the permutation null are grouped by patient. MIMIC-CXR has 1,491 subjects with one image each.

**Which CheXpert arrays to use.** The revised paper (CMPB revision, 2026) uses ResNet-50, DenseNet-121 and ViT-B/16 from `chexpert_multibackbone/<backbone>_embeddings.npy`, and ViT-B/32, DINOv2, RAD-DINO and MedSigLIP-448 from `chexpert_multibackbone/public/`. The `public/` arrays were extracted from the same 2,000 images with [`scripts/revision/extract_missing_chexpert_backbones.py`](https://github.com/criticaldata/ShortKit-ML/blob/main/scripts/revision/extract_missing_chexpert_backbones.py) and pinned public checkpoints (torchvision ViT-B/32 default weights, `facebook/dinov2-base`, `microsoft/rad-dino`, `google/medsiglip-448`); rows follow `chexpert_multibackbone/<backbone>_metadata.csv`. The same code reproduces the stored ViT-B/16 array to cosine 1.000. The older ViT-B/32, DINOv2, RAD-DINO and MedSigLIP arrays have no extraction code and are kept only to replay the first submission. The MIMIC-CXR array labelled MedSigLIP is `google/medsiglip-448` vision pooler output: re-embedding 50 of its images reproduces it at cosine 1.000, although its source README names `google/siglip-so400m-patch14-384`.

To regenerate every table and figure of the revised paper: `bash scripts/revision/run_all.sh`.

### Requesting access

The dataset is gated because it holds data derived from MIMIC-CXR, which PhysioNet releases only to credentialed users. To get access:

1. Click **Request access** on the [dataset page](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data) and fill in the form, including your PhysioNet username.
2. Email your PhysioNet credentials to **sebasmos@mit.edu**: your PhysioNet username and proof of current MIMIC-CXR access (for example a screenshot of the MIMIC-CXR project page showing you have access, or your CITI training certificate). Never send your PhysioNet password.
3. Access is granted by hand once the credentials are checked.

CheXpert users must also have accepted the Stanford CheXpert research use agreement.

### Paper tables and figures

The revised paper's tables (`.tex`), figures (`.pdf`) and `numbers.tex` are written by the `tables` stage of `scripts/revision/run_all.sh` to `output/paper_tables/revision/`, with a `MANIFEST.txt` recording the commit, data hashes and settings. See `docs/reproducibility.md` for the stage list.

## GPU Support

The library auto-selects the best available device. PyTorch components (probes, VAE, GroupDRO, adversarial debiasing, etc.) use the standard `torch.device` fallback:

| Platform | Backend | Auto-detected |
|----------|---------|---------------|
| Linux/Windows with NVIDIA GPU | CUDA | Yes (`torch.cuda.is_available()`) |
| macOS Apple Silicon | MPS | Partial -- pass `device="mps"` explicitly |
| CPU (any platform) | CPU | Yes (default fallback) |

> **Note:** Most detection methods (HBAC, statistical, geometric, etc.) run on CPU via NumPy/scikit-learn and do not require GPU. GPU acceleration benefits the torch-based probe, VAE, GroupDRO, and mitigation methods. MPS support depends on PyTorch operator coverage; if you encounter errors on Apple Silicon, fall back to `device="cpu"`.

## Interactive Dashboard

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

Features: sample CheXpert data, custom CSV upload, PDF/HTML reports, model comparison tab, multi-attribute analysis.

**CSV Format:**
```csv
embedding_0,embedding_1,...,task_label,group_label,attr_race,attr_gender
0.123,0.456,...,1,group_a,Black,Male
```

> See [Dashboard Guide](docs/getting-started/dashboard.md) for detailed usage.

## Testing

```bash
pytest tests/ -v
pytest --cov=shortcut_detect --cov-report=html
```

The test suite covers all detection and mitigation methods.

## Contributing

```bash
pip install -e ".[dev]"
pre-commit install
```

- **Black** for formatting (line length: 100), **Ruff** for linting, **MyPy** for types
- Pre-commit hooks run automatically; CI tests on Python 3.10, 3.11, 3.12
- New detectors must implement `DetectorBase`. See `docs/contributing.md` and `shortcut_detect/detector_template.py`

## Project Structure

```
shortcut_detect/
├── probes/           # Probe-based detection (sklearn + torch)
├── clustering/       # HBAC detector
├── statistical/      # Statistical testing
├── geometric/        # Geometric & bias direction analysis
├── fairness/         # Equalized Odds, Demographic Parity, Intersectional
├── frequency/        # Frequency shortcut detector
├── causal/           # Causal effect detector
├── gce/              # Generalized cross-entropy detector
├── training/         # Early epoch clustering (SPARE)
├── vae/              # VAE latent disentanglement
├── xai/              # CAV, SpRAy, GradCAM mask overlap, SIS
├── ssa/              # Semi-supervised spectral analysis
├── groupdro/         # GroupDRO worst-group robustness
├── conditions/       # Pluggable risk aggregation conditions
│   ├── base.py, registry.py, indicator_count.py, majority_vote.py
│   ├── weighted_risk.py, multi_attribute.py, meta_classifier.py
│   └── meta_model.joblib  # Trained meta-classifier (bundled)
├── benchmark/        # Paper benchmark infrastructure
│   ├── runner.py, paper_runner.py, synthetic_generator.py
│   ├── measurement.py, fp_analysis.py, sensitivity.py
│   ├── convergence_viz.py, baseline_comparison.py, figures.py
├── comparison/       # Model comparison runner
├── mitigation/       # Debiasing & masking methods (M01-M07)
├── reporting/        # HTML/PDF/CSV reports & visualizations
├── unified.py        # ShortcutDetector unified API
└── detector_base.py  # DetectorBase ABC with results_ schema

docs/                 # MkDocs documentation site
examples/             # Notebooks and benchmark configs
app.py                # Gradio dashboard
Dockerfile            # Reproducible environment
scripts/              # Paper reproduction scripts
tests/                # Test suite
```

## Documentation

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]" pymdown-extensions
mkdocs serve  # http://127.0.0.1:8000
```

- [Getting Started](docs/getting-started/installation.md)
- [Detection Methods](docs/methods/overview.md) -- all 20+ methods with guides
- [API Reference](docs/api/shortcut-detector.md)
- [Contributing](docs/contributing.md)

## Citation

The paper is under revision at *Computer Methods and Programs in Biomedicine*. Preprint:

```bibtex
@article{cajas2026shortkit,
  title={ShortKit-ML: an open toolkit for measuring shortcut potential in medical imaging embeddings},
  author={Cajas, Sebastian and Marzullo, Aldo and Kapadia, Sahil and Santos, Filipe and Ocampo Osorio, Felipe and Kong, Qingpeng and Quarta, Alessandro and Kuo, Po-Chih and Patel, Milit and Rojas Sillery, Raul Ignacio and Celi, Leo Anthony},
  journal={medRxiv},
  year={2026},
  doi={10.64898/2026.04.29.26352053},
  url={https://www.medrxiv.org/content/10.64898/2026.04.29.26352053v1}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contact

- **GitHub**: [criticaldata/ShortKit-ML](https://github.com/criticaldata/ShortKit-ML)
- **Issues**: [GitHub Issues](https://github.com/criticaldata/ShortKit-ML/issues)
