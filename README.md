# ShortKit-ML

<p align="center">
  <img src="docs/assets/shortkit.png" alt="ShortKit-ML Logo" width="400"/>
</p>

> **ShortKit-ML** — Detect and mitigate shortcuts and biases in machine learning embedding spaces. 20+ detection and mitigation methods with a unified API. **Multi-attribute support** tests multiple sensitive attributes simultaneously. Model Comparison mode for benchmarking multiple embedding models.

[![PyPI version](https://img.shields.io/pypi/v/shortkit-ml.svg)](https://pypi.org/project/shortkit-ml/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![CI](https://github.com/criticaldata/ShortKit-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/criticaldata/ShortKit-ML/actions/workflows/tests.yml)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-ShortKIT--ML--data-yellow.svg)](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data)
[![Docs](https://img.shields.io/badge/docs-criticaldata.github.io-blue.svg)](https://criticaldata.github.io/ShortKit-ML/)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detection Methods](#detection-methods)
- [Overall Assessment Conditions](#overall-assessment-conditions)
- [MCP Server](#mcp-server)
- [Paper Benchmarks](#paper-benchmark-datasets)
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

> See [Embedding-Only Guide](docs/methods/overview.md) for `CallableEmbeddingSource` and caching options.

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

ShortKit-ML ships an [MCP](https://modelcontextprotocol.io/) server so AI assistants (Claude, Cursor, etc.) can call detection tools directly from chat — no Python script required.

### Install the MCP extra

```bash
pip install -e ".[mcp]"
```

### Start the server

```bash
# via entry point (after install)
shortkit-ml-mcp

# or directly
python -m shortcut_detect.mcp_server
```

### Available tools

| Tool | Description |
|------|-------------|
| `list_methods` | List all 19 detection methods with descriptions |
| `generate_synthetic_data` | Generate a synthetic shortcut dataset (linear / nonlinear / none) |
| `run_detector` | Run selected methods on embeddings — returns verdict, risk level, per-method breakdown |
| `get_summary` | Human-readable summary from a prior `run_detector` call |
| `get_method_detail` | Full raw result dict for a single method |
| `compare_methods` | Side-by-side comparison table + consensus vote across methods |

### Connect to Claude Desktop

Add the following to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "shortkit-ml": {
      "command": "python",
      "args": ["-m", "shortcut_detect.mcp_server"],
      "cwd": "/path/to/ShortKit-ML"
    }
  }
}
```

A ready-to-edit template is included at [`claude_desktop_config.json`](claude_desktop_config.json).

## Paper Benchmark Datasets

### Dataset 1 -- Synthetic Grid

Configure `examples/paper_benchmark_config.json` to control effect sizes, sample sizes, imbalance ratios, and embedding dimensionalities. A smoke profile (`examples/paper_benchmark_config_smoke.json`) is provided for quick sanity checks.

```bash
python -m shortcut_detect.benchmark.paper_run --config examples/paper_benchmark_config.json
```

Outputs CSVs, figures, and summary markdown into `output/paper_benchmark/`.

### Dataset 2 -- CheXpert Real Data

Requires a CheXpert manifest (`data/chexpert_manifest.csv`) plus model-specific embedding pickles. Supported models: `medclip`, `biomedclip`, `cxr-foundation`.

```bash
python3 scripts/run_dataset2_benchmark.py \
  --manifest data/chexpert_manifest.csv \
  --model medclip \
  --root . \
  --artifacts-dir output/paper_benchmark/chexpert_embeddings \
  --config examples/paper_benchmark_config.json
```

See `scripts/reproduce_paper.sh` and the Dockerfile for full reproducibility.

## Reproducing Paper Results

All paper results are fully reproducible with fixed seeds (`seed=42`). Every table and figure in the paper can be regenerated from the scripts and data in this repository.

**13 benchmark methods** are evaluated across all datasets: `hbac`, `probe`, `statistical`, `geometric`, `frequency`, `bias_direction_pca`, `sis`, `demographic_parity`, `equalized_odds`, `intersectional`, `groupdro`, `gce`, `ssa`. These span 5 paradigms: embedding-level analysis, representation geometry, fairness evaluation, explainability, and training dynamics.

### Step-by-step Reproduction

| Step | Command | Output | Time |
|------|---------|--------|------|
| 1. Install | `pip install -e ".[all]"` | Package + deps | 2 min |
| 2. Synthetic benchmarks | `python scripts/generate_all_paper_tables.py` | `output/paper_tables/*.tex` | ~10 min |
| 3. Paper figures | `python scripts/generate_paper_figures.py` | `output/paper_figures/*.pdf` | ~2 min |
| 4. CheXpert benchmark | `python scripts/run_chexpert_benchmark.py` | `output/paper_benchmark/chexpert_results/` | ~1 min |
| 5. MIMIC-CXR setup | `python scripts/setup_mimic_cxr_data.py` | `data/mimic_cxr/*.npy` | ~1 min |
| 6. MIMIC-CXR benchmark | `python scripts/run_mimic_benchmark.py` | `output/paper_benchmark/mimic_cxr_results/` | ~2 min |
| 7. CelebA extraction | `python scripts/extract_celeba_embeddings.py` | `data/celeba/celeba_real_*.npy` | ~5 min (MPS) |
| 8. CelebA benchmark | `python scripts/run_celeba_real_benchmark.py` | `output/paper_benchmark/celeba_real_results/` | ~1 min |
| 9. Full pipeline (smoke) | `./scripts/reproduce_paper.sh smoke` | All synthetic outputs | ~5 min |
| 10. Full pipeline | `./scripts/reproduce_paper.sh full` | All synthetic outputs | ~2-4 hrs |

### Docker (fully self-contained)
```bash
docker build -t shortcut-detect .
docker run --rm -v $(pwd)/output:/app/output shortcut-detect full
```

### Data

> **Important:** The `data/` folder in this repository is empty. All embeddings and metadata are hosted on HuggingFace and must be downloaded separately. Raw CheXpert and MIMIC-CXR images and labels are **not redistributed** — access requires accepting the respective dataset licenses (PhysioNet for MIMIC-CXR, Stanford for CheXpert).

All embeddings are hosted on HuggingFace (gated, PhysioNet-restricted access):

**[MITCriticalData/ShortKit-ML-data](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data)**

```bash
# Download all embeddings into data/
huggingface-cli download MITCriticalData/ShortKit-ML-data --repo-type dataset --local-dir data/
```

| Dataset | Location | Embedding Models | Dim | Samples |
|---------|----------|-----------------|-----|---------|
| Synthetic | Generated at runtime | `SyntheticGenerator(seed=42)` | 128 | Configurable |
| CheXpert | `data/chexpert/` | MedCLIP, ResNet-50, DenseNet-121, ViT-B/16, ViT-B/32, DINOv2, RAD-DINO, MedSigLIP | 512-2048 | 2,000 each |
| MIMIC-CXR | `data/mimic_cxr/` | RAD-DINO, ViT-B/16, ViT-B/32, MedSigLIP | 768-1152 | ~1,500 each |
| CelebA | `data/celeba/` | ResNet-50 (ImageNet) | 2,048 | 10,000 |

### Paper Tables → Scripts Mapping

| Paper Table | Script | Data | Seed |
|-------------|--------|------|------|
| Tab 3: Synthetic P/R/F1 | `generate_all_paper_tables.py` | `SyntheticGenerator` | 42 |
| Tab 4: False positive rates | `generate_all_paper_tables.py` | `SyntheticGenerator` (null) | 42 |
| Tab 5: Sensitivity analysis | `generate_all_paper_tables.py` | `SensitivitySweep` | 42 |
| Tab 6: CheXpert results | `run_chexpert_benchmark.py` | `data/chest_embeddings.npy` | 42 |
| Tab 7: MIMIC-CXR cross-val | `run_mimic_benchmark.py` | `data/mimic_cxr/*.npy` | 42 |
| Tab 8: CelebA validation | `run_celeba_real_benchmark.py` | `data/celeba/celeba_real_embeddings.npy` | 42 |
| Tab 9: Risk conditions | `generate_all_paper_tables.py` | `SyntheticGenerator` | 42 |
| Fig 2: Convergence matrix | `generate_paper_figures.py` | Synthetic + CheXpert | 42 |

See `docs/reproducibility.md` for full details.

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

**638 tests passing** across all detection and mitigation methods.

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
tests/                # Test suite (475+ tests)
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

```bibtex
@software{shortkit_ml2025,
  title={ShortKit-ML: Tools for Identifying Biases in Embedding Spaces},
  author={Sebastian Cajas, Aldo Marzullo, Sahil Kapadia, Qingpeng Kong, Filipe Santos, Alessandro Quarta, Leo Celi},
  year={2025},
  url={https://github.com/criticaldata/ShortKit-ML}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contact

- **GitHub**: [criticaldata/ShortKit-ML](https://github.com/criticaldata/ShortKit-ML)
- **Issues**: [GitHub Issues](https://github.com/criticaldata/ShortKit-ML/issues)
