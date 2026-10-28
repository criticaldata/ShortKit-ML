# Reproducing Paper Results

This guide explains how to reproduce the benchmark results from the ShortKit-ML paper.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- pip or uv package manager
- (Optional) Docker for fully isolated runs

## Quick Start

### 1. Install the package

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd Shortcut_Detect
pip install -e ".[dev]"
```

### 2. Run the reproducibility script

```bash
# Quick sanity check
./scripts/reproduce_paper.sh smoke

# Moderate run (recommended first attempt)
./scripts/reproduce_paper.sh default

# Full paper reproduction
./scripts/reproduce_paper.sh full
```

## Profiles

| Profile   | Grid size | Seeds | Expected runtime  |
|-----------|-----------|-------|-------------------|
| `smoke`   | 2 effect sizes, 1 sample size, 1 dim | 2 | 2-5 minutes |
| `default` | 4 effect sizes, 2 sample sizes, 2 dims | 3+ | 15-30 minutes |
| `full`    | 5 effect sizes, 3 sample sizes, 3 dims | 10 | 2-4 hours |

All profiles use `random_seed: 42` for deterministic results.

## Using Docker

For a fully isolated, reproducible environment:

```bash
# Build the image
docker build -t shortcut-detect .

# Run with a specific profile (output is mounted to the host)
docker run --rm -v $(pwd)/output:/app/output shortcut-detect smoke
docker run --rm -v $(pwd)/output:/app/output shortcut-detect default
docker run --rm -v $(pwd)/output:/app/output shortcut-detect full
```

## Configuration

The reproducible config is at `examples/paper_benchmark_config_reproducible.json`. It explicitly specifies every parameter so there is no ambiguity:

- **random_seed**: 42 (fixed for all runs)
- **methods**: hbac, probe, statistical, geometric
- **effect_sizes**: 0.2, 0.5, 0.8, 1.2, 2.0 (Cohen's d)
- **sample_sizes**: 200, 1000, 5000
- **imbalance_ratios**: 0.5 (balanced), 0.9 (imbalanced)
- **embedding_dims**: 128, 256, 512
- **shortcut_dims**: 5
- **seeds**: 10 independent seeds per configuration
- **alpha**: 0.05
- **corrections**: bonferroni, fdr_bh

For custom runs, copy the config and modify as needed, then call:

```bash
python -m shortcut_detect.benchmark.paper_runner --config your_config.json
```

## Expected Output Files

After a successful run, the timestamped output directory contains:

| File | Description |
|------|-------------|
| `synthetic_runs.csv` | Raw results for every synthetic configuration |
| `synthetic_power_recall.csv` | Power/recall analysis across effect sizes |
| `synthetic_false_positive.csv` | False positive rates under the null |
| `correction_comparison.csv` | Multiple testing correction comparison |
| `benchmark_meta.json` | Run metadata (config, timing, environment) |

## CheXpert (Dataset 2)

The CheXpert real-data benchmark requires access to the CheXpert dataset. Set `chexpert.enabled: true` and provide `chexpert.manifest_path` in the config. See `scripts/extract_chexpert_embeddings.py` for embedding extraction.

## Revised paper (CMPB revision, 2026)

Every table, figure and number of the revised paper comes from one pipeline:

```bash
./scripts/reproduce_paper.sh revision          # all stages
scripts/revision/run_all.sh head mlp tables    # selected stages
```

Stages, in order: `old clinical negctl synthetic nullcheck head mlp analyses umap marker extraction alignment overlap sensitivity tables`. Results go to `~/Documents/shortkit-ml-results/2026-10-28_revision_<commit>/` unless `SHORTKIT_REVISION_DIR` is set; each stage writes a `MANIFEST.txt` with the commit, data hashes and settings. `tables` writes the LaTeX tables, figures and `numbers.tex` to `output/paper_tables/revision/`.

The embeddings come from the gated dataset [MITCriticalData/ShortKit-ML-data](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data). To get access, click **Request access** on the dataset page, then email your PhysioNet username and proof of current MIMIC-CXR access (never your password) to sebasmos@mit.edu. Requests are approved by hand.

For CheXpert the pipeline reads ResNet-50, DenseNet-121 and ViT-B/16 from `data/chexpert_multibackbone/`, and ViT-B/32, DINOv2, RAD-DINO and MedSigLIP-448 from `data/chexpert_multibackbone/public/`. The `extraction` stage regenerates the `public/` arrays from the CheXpert images with pinned public checkpoints (it needs torch 2.6 or later with `transformers`, set through `EXTRACT_PYTHON`, and accepted terms for `google/medsiglip-448`).

## Troubleshooting

- **Import errors**: Make sure the package is installed with `pip install -e ".[dev]"`.
- **PyTorch/CUDA**: For GPU support, install PyTorch separately following [pytorch.org](https://pytorch.org/get-started/locally/).
- **PDF export errors**: Install system libraries for weasyprint: `brew install pango gdk-pixbuf libffi` (macOS) or `apt-get install libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev` (Debian/Ubuntu).
