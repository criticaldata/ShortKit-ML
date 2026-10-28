# Benchmark Methods: Decision Log & Reproducibility Guide

This document records the method selection of the **first submission** (13 methods compared against fixed thresholds) and how to replay it. The revised paper scores six calibrated detectors instead, with the conditional probe as the single primary test; see [Results](#results) below and [Paper Results](reproducibility.md#revised-paper-cmpb-revision-2026).

## What the revised paper scores

Of the 20 implemented detection methods, six need only the embedding, the attribute and the task label and are calibrated by the conditional permutation null: the conditional probe (primary) and the statistical, geometric, bias direction PCA, HBAC and frequency detectors (secondary, same null, reported as diagnostics). The number of flagged detectors in a cell is descriptive only; no agreement threshold is used. SIS is outside the score (it fits on the task label with a fixed 15% subset and has no attribute-aware form). Demographic parity, equalized odds and intersectional gaps are computed only for the trained heads of the classification-head experiment, and GroupDRO, GCE, SSA, early-epoch clustering, CAV, SpRAy, GradCAM, causal effect, the CVAE and the VAE contribute to no result in the paper.

## First-submission method selection

Of the 19 detection methods counted in the first submission, **13 were benchmarked** there. The remaining 6 require raw images, concept sets, or specialized data loaders that are incompatible with the embedding-only evaluation protocol.

### Benchmarked Methods (13, first submission)

| # | Method | Key | Tier | Paradigm | What It Detects | Detection Logic |
|---|--------|-----|------|----------|-----------------|-----------------|
| 1 | HBAC | `hbac` | 1 | Embedding | Clustering by protected attributes | `report.has_shortcut.exists == True` |
| 2 | Probe | `probe` | 1 | Embedding | Group info recoverable from embeddings | `results.shortcut_detected == True` (F1 > threshold) |
| 3 | Statistical | `statistical` | 1 | Embedding | Dimensions with group differences | Any significant features after FDR-BH correction |
| 4 | Geometric | `geometric` | 1 | Representation | Bias directions & prototype overlap | `summary.risk_level ∈ {moderate, high}` |
| 5 | Frequency | `frequency` | 1 | Embedding | Signal concentrated in few dimensions | `report.shortcut_detected == True` (TPR/FPR thresholds) |
| 6 | Bias Direction PCA | `bias_direction_pca` | 1 | Representation | Projection gap along bias direction | `report.risk_level ∈ {moderate, high}` |
| 7 | SIS | `sis` | 1 | Explainability | Minimal dimensions sufficient for prediction | `shortcut_detected == True` (frac_dims ≤ 0.15) |
| 8 | Demographic Parity | `demographic_parity` | 2 | Fairness | Prediction rate disparities across groups | `report.risk_level ∈ {moderate, high}` |
| 9 | Equalized Odds | `equalized_odds` | 2 | Fairness | TPR/FPR disparities across groups | `report.risk_level ∈ {moderate, high}` |
| 10 | Intersectional | `intersectional` | 2 | Fairness | Intersectional fairness gaps (2+ attributes) | `report.risk_level ∈ {moderate, high}` |
| 11 | GroupDRO | `groupdro` | 3 | Training Dynamics | Worst-group performance gaps | `avg_acc - worst_group_acc > 0.10` |
| 12 | GCE | `gce` | 3 | Training Dynamics | High-loss minority/bias-conflicting samples | `report.risk_level ∈ {moderate, high}` |
| 13 | SSA | `ssa` | 3 | Training Dynamics | Spurious attribute spread | `shortcut_detected == True` |

### Tiers Explained

| Tier | Name | Requirements | Methods |
|------|------|-------------|---------|
| **1** | Embedding-native | embeddings + labels + group_labels | hbac, probe, statistical, geometric, frequency, bias_direction_pca, sis |
| **2** | Fairness-based | embeddings + binary labels + protected_labels | demographic_parity, equalized_odds, intersectional |
| **3** | Training-dynamics | embeddings + labels + group_labels (trains internal model) | groupdro, gce, ssa |

**All 13 methods** work with the `ShortcutDetector.fit(embeddings, labels, group_labels=g)` API. No raw images, model weights, or external data required.

### Excluded Methods (6) and Why

| Method | Key | Why Excluded | Could Be Included? |
|--------|-----|-------------|-------------------|
| CAV | `cav` | Requires concept activation sets via `fit_from_loaders()` | Only with manually curated concept sets |
| GradCAM Mask Overlap | `gradcam_mask_overlap` | Requires raw images + model with gradients | Only with image data + model access |
| Early Epoch Clustering | `early_epoch_clustering` | Requires epoch-by-epoch representations during training | Only during model training |
| VAE | `vae` | Trains a VAE; expensive and noisy on small samples | Possible but slow (~minutes per run on 2K samples) |
| Generative CVAE | `generative_cvae` | Trains a CVAE counterfactual generator | Possible but very slow |
| Causal Effect | `causal_effect` | Requires interventional data via `fit_from_loaders()` | Only with causal/interventional data |

### Special Cases

- **Intersectional**: Requires `extra_labels` with 2+ demographic attributes. Skipped in synthetic benchmarks (1 attribute only) and in CheXpert per-diagnosis (single attribute tested). Works on MIMIC-CXR (sex + race + age_bin) and CelebA (40 attributes).
- **GroupDRO**: Detection threshold is `avg_acc - worst_group_acc > 0.10` (10% gap). In practice, GroupDRO reports `success=False` on most benchmark runs because its internal training doesn't converge or produce a meaningful gap with pre-extracted embeddings. It was designed for end-to-end training scenarios. **Effectively 0 detections in benchmarks.**
- **GCE**: Reports `risk=high` on most runs but doesn't meet the flagging threshold via `method_flag()` because its `risk_level` in the report dict stays at "low". GCE is designed to identify high-loss minority samples; on pre-extracted embeddings with binary labels, the loss landscape is shallow. **Effectively 0 detections in benchmarks.**
- **SSA**: Requires a `splits` parameter (labeled/unlabeled split indices) that the benchmark scripts don't provide. Fails with "splits parameter required for SSA analysis". **Effectively 0 detections in benchmarks.**
- **Net effect of Tier 3 methods**: GroupDRO, GCE, and SSA are included for completeness (the Gradio demo runs them) but they add 0 detections in the embedding-only benchmark protocol. The effective detection count is X/13 where the 3 Tier 3 methods always report "not flagged." This is documented honestly: these methods were designed for training-time scenarios, not post-hoc embedding analysis.

## Datasets & Backbones

| Dataset | Backbones | Samples | Attributes | Methods Run |
|---------|-----------|---------|------------|-------------|
| **Synthetic** | N/A (128-dim generated) | 200-5000 | 1 (binary) | 12 (no intersectional) |
| **CheXpert** | ResNet-50, DenseNet-121, ViT-B/16, ViT-B/32, DINOv2, RAD-DINO, MedSigLIP-448 | 2,000 each | sex, age_bin, race | 13 |
| **MIMIC-CXR** | RAD-DINO, ViT-B/16, ViT-B/32, SigLIP So400m (files labelled MedSigLIP) | 1,491 each | race, sex, age_bin | 13 |
| **CelebA** | ResNet-50 | 10,000 | Male + 39 binary attrs | 13 |
| **Cross-dataset** | ResNet-50 (CheXpert) + RAD-DINO (MIMIC) | varies by diagnosis | sex, race | 13 (intersectional fails on single-attr CheXpert) |

## Replaying the first-submission benchmark

The commands below rerun the 13-method protocol. The `old` stage of `scripts/revision/run_all.sh` reruns its clinical tables with the fixed flag reader for the comparison in the revised paper.

### Prerequisites

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git
cd Shortcut_Detect
pip install -e ".[all]"

# Download embeddings (gated: request access on the dataset page, then email your
# PhysioNet username and proof of MIMIC-CXR access to sebasmos@mit.edu)
huggingface-cli download MITCriticalData/ShortKit-ML-data --repo-type dataset --local-dir data/
```

### Run Benchmarks (one at a time for best performance)

```bash
# 1. CheXpert: 7 backbones × 3 attributes × 13 methods + per-diagnosis
python scripts/run_chexpert_multibackbone_benchmark.py
# Output: output/paper_benchmark/chexpert_multibackbone_results/

# 2. MIMIC-CXR: 4 backbones × 3 attributes × 13 methods
python scripts/run_mimic_benchmark.py
# Output: output/paper_benchmark/mimic_cxr_results/

# 3. CelebA: 3 shortcut pairs × 13 methods
python scripts/run_celeba_real_benchmark.py
# Output: output/paper_benchmark/celeba_real_results/

# 4. Cross-dataset per-diagnosis: 5 diagnoses × 3 conditions × 13 methods
python scripts/run_per_diagnosis_cross_dataset.py
# Output: output/paper_benchmark/cross_dataset_diagnosis/

# 5. Synthetic: 12 methods × effect size grid (smoke profile ~5 min)
python -m shortcut_detect.benchmark.paper_runner --config examples/paper_benchmark_config_reproducible.json
# Output: output/paper_benchmark/synthetic_10methods/
```

### Expected Runtimes (Apple M-series, single process)

| Benchmark | Methods | Combos | Approx. Time |
|-----------|---------|--------|-------------|
| MIMIC-CXR | 13 | 12 | ~4 min |
| CheXpert (attrs only) | 13 | 21 | ~15 min |
| CheXpert (+ per-diagnosis) | 13 | 21 + 35 | ~30 min |
| CelebA | 13 | 3 | ~30 min (40 extra_labels) |
| Cross-dataset | 13 | 15 | ~4 hrs (geometric bottleneck) |
| Synthetic (smoke) | 12 | ~100 | ~5 min |
| Synthetic (full) | 12 | ~12K | ~2-4 hrs |

### Reproducibility Guarantees

- **Seed**: All runs use `seed=42`
- **Data**: Pre-extracted embeddings on HuggingFace (no GPU needed)
- **Determinism**: NumPy/scikit-learn random states fixed per method
- **CI**: the test suite runs on Python 3.10, 3.11 and 3.12 (GitHub Actions)

## Results

The 13-method detection-rate summary from the first submission is superseded. The revised paper reports a conditional audit (the conditional probe as primary test, five secondary detectors sharing its permutation null, BH correction within a cell) instead; its tables and figures are built by the pipeline in [Paper Results](reproducibility.md#revised-paper-cmpb-revision-2026) and written to `output/paper_tables/revision/`, with the resolved numbers in `numbers.json` there.

## File Locations

| File | Purpose |
|------|---------|
| `shortcut_detect/benchmark/method_utils.py` | Canonical method list, `method_flag()`, `convergence_bucket()` |
| `shortcut_detect/benchmark/paper_runner.py` | Synthetic benchmark orchestrator |
| `scripts/run_chexpert_multibackbone_benchmark.py` | CheXpert benchmark |
| `scripts/run_mimic_benchmark.py` | MIMIC-CXR benchmark |
| `scripts/run_celeba_real_benchmark.py` | CelebA benchmark |
| `scripts/run_per_diagnosis_cross_dataset.py` | Cross-dataset per-diagnosis |
| `tests/test_method_utils.py` | Unit tests for method detection logic |
| `docs/benchmark_methods.md` | This file |
