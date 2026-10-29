# Reproducing Paper Results

This guide explains how to reproduce the results of the ShortKit-ML paper (under revision at *Computer Methods and Programs in Biomedicine*; [preprint](https://www.medrxiv.org/content/10.64898/2026.04.29.26352053v1)). The revised paper's pipeline is in [Revised paper](#revised-paper-cmpb-revision-2026) below. The `smoke`, `default` and `full` profiles in the next sections run the first-submission synthetic benchmark (fixed-threshold detectors) and do not produce the revised paper's tables.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- pip or uv package manager
- (Optional) Docker for fully isolated runs

## Quick Start

### 1. Install the package

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd ShortKit-ML
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

Stages, in order: `old clinical negctl synthetic nullcheck head dfr mlp race analyses umap marker extraction alignment overlap sensitivity sample_vs_full nlprobe nlprobe_clinical scaling tables`. The `race` stage repeats the head experiment with race (White against non-White) as the attribute, logistic head only. The `sensitivity` stage runs the probe checks `C`, `frontal` and `labelled`. The `sample_vs_full` stage compares the audited CheXpert sample with the full `train.csv` on sex, age, view and race. The `nlprobe` stage runs the conditional probe on squared and pairwise-product features of the synthetic embeddings, under the same permutation null, at one small setting (`nonlinear_probe/`). The `nlprobe_clinical` stage fits the probe on `[E, E^2]` for every primary cell of the clinical audit (CheXpert and MIMIC-CXR, not CelebA) with the audit's seed, K = 500, patient-grouped folds and permutation draws, and joins the linear probe's AUROC and decision from `clinical_audit/` (`nonlinear_probe_clinical/`); the pairwise-product design is not run at d up to 2048. The `scaling` stage times the probe alone (K = 500, patient-grouped folds) on synthetic 768-dimensional embeddings at 2,000 to 50,000 images and fits a log-log slope (`probe_scaling/`). One more stage runs only when named: `prepare_mimic` rebuilds `data/mimic_cxr/vit32_cls_embeddings_realigned.npy` and `mimic_race_raw.csv` from the qml-mimic-cxr-embeddings parquet files (path through `QML_DIR`; needs `pyarrow`). The `Stored array` column of the sensitivity table reads `clinical_audit_stored_arrays/`, which no stage writes: it is a frozen output of the `clinical` stage from before the CheXpert switch to the `public/` arrays. Results go to `~/Documents/shortkit-ml-results/2026-10-28_revision_<commit>/` unless `SHORTKIT_REVISION_DIR` is set; each stage writes a `MANIFEST.txt` with the commit, data hashes and settings. `tables` writes the LaTeX tables, figures and `numbers.tex` to `output/paper_tables/revision/`.

The embeddings come from the gated dataset [MITCriticalData/ShortKit-ML-data](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data). To get access, click **Request access** on the dataset page, then email your PhysioNet username and proof of current MIMIC-CXR access (never your password) to sebasmos@mit.edu. Requests are approved by hand.

The audit in every stage is `shortcut_detect.audit.conditional_audit`: the conditional probe (L2 logistic regression predicting the attribute from the embedding and the task label, out-of-fold AUROC) is the single primary test, calibrated by permuting the attribute within task-label strata and between patients (K=500 in the clinical audits, 200 for the secondary detectors, the synthetic grid and the audits after mitigation). The five secondary detectors share that null; the six p-values of a cell are Benjamini-Hochberg adjusted, and the count of flagged detectors is descriptive only. The audit reports shortcut potential; the classification-head experiment (`head`, `mlp`, `race`, `dfr` stages) tests separately whether a head trained on confounded data uses the attribute.

For CheXpert the pipeline reads ResNet-50, DenseNet-121 and ViT-B/16 from `data/chexpert_multibackbone/`, and ViT-B/32, DINOv2, RAD-DINO and MedSigLIP-448 from `data/chexpert_multibackbone/public/`. The `extraction` stage regenerates the `public/` arrays from the CheXpert images with pinned public checkpoints (it needs torch 2.6 or later with `transformers`, set through `EXTRACT_PYTHON`, and accepted terms for `google/medsiglip-448`). The CheXpert sample is the first 2,000 images in the order of the official `train.csv` (477 patients, 423 lateral views); sex comes from `train.csv`, race per patient from the demographics file (White against non-White in the primary analysis, 217 images without a usable race record left out), and the task is Pneumothorax. MIMIC-CXR has 1,491 subjects with one image each, task in-hospital mortality, with four backbones: RAD-DINO, ViT-B/16, ViT-B/32 (rebuilt from the source in metadata order) and the array labelled MedSigLIP-448 at its source, which was extracted with `google/siglip-so400m-patch14-384` and is reported as SigLIP (So400m).

## Troubleshooting

- **Import errors**: Make sure the package is installed with `pip install -e ".[dev]"`.
- **PyTorch/CUDA**: For GPU support, install PyTorch separately following [pytorch.org](https://pytorch.org/get-started/locally/).
- **PDF export errors**: Install system libraries for weasyprint: `brew install pango gdk-pixbuf libffi` (macOS) or `apt-get install libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev` (Debian/Ubuntu).
