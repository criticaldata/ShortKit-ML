---
license: cc-by-nc-4.0
extra_gated_heading: "Acknowledge license and PhysioNet data use agreement"
extra_gated_description: "This dataset contains derived data from PhysioNet restricted-access datasets (MIMIC-CXR). By requesting access, you confirm that you have an active PhysioNet credentialed account and have signed the relevant data use agreements."
extra_gated_button_content: "Request access"
extra_gated_prompt: "You agree to not use this dataset to conduct experiments that cause harm to human subjects, and you confirm compliance with the PhysioNet data use agreement."
extra_gated_fields:
  Full Name: text
  Affiliation: text
  Country: country
  PhysioNet Username: text
  I want to use this dataset for:
    type: select
    options:
      - Research
      - Education
      - label: Other
        value: other
  I have a valid PhysioNet credentialed account with MIMIC-CXR access: checkbox
  I agree to use this dataset for non-commercial use ONLY: checkbox
tags:
- medical-imaging
- chest-xray
- embeddings
- shortcut-detection
- fairness
- bias-detection
- celeba
- chexpert
- mimic-cxr
---

# ShortKit-ML Benchmark Data

Pre-computed embeddings, metadata, and **full original dataset labels** for reproducing paper benchmarks. All embeddings were extracted with `seed=42` for full reproducibility.

## Full Dataset Files (not just embeddings)

This repository includes the **complete original label/metadata files** for CheXpert and MIMIC-CXR — not only the embedding subsets used in our experiments:

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 223,414 | **Full CheXpert training set** — Path, Sex, Age, AP/PA, 14 diagnosis labels |
| `valid.csv` | 234 | **Full CheXpert validation set** — same schema |
| `mimic_cxr/mimic-cxr-2.0.0-chexpert.csv` | 227,827 | **Full MIMIC-CXR diagnosis labels** — 14 CheXpert-style labels per study |
| `mimic_cxr/mimic-cxr-2.0.0-metadata.csv` | 377,110 | **Full MIMIC-CXR DICOM metadata** — view position, rows, cols, study date |
| `chexpert_multibackbone/race_mapping.csv` | — | CheXpert patient-to-race mapping (from CHEXPERT DEMO) |

These are the same files distributed by Stanford (CheXpert) and PhysioNet (MIMIC-CXR). No rows have been filtered or removed.

## Embedding Subsets (for benchmark reproduction)

The embedding files below are **subsets** extracted for our experiments (2,000 CheXpert samples, 1,491 MIMIC-CXR samples, 10,000 CelebA samples).

```
data/
├── chest_embeddings.npy          # CheXpert MedCLIP embeddings (2000, 512)
├── chest_labels.npy              # Binary task labels (2000,)
├── chest_group_labels.npy        # Race groups: 0=ASIAN,1=BLACK,2=OTHER,3=WHITE
├── chexpert_manifest.csv         # CheXpert metadata (image_path, task_label, race, sex, age)
│
├── chexpert/                     # CheXpert 8 backbones (from danjacobellis/chexpert)
│   ├── {backbone}_embeddings.npy # 8 backbones × 2000 samples each
│   ├── {backbone}_metadata.csv   # sex, age, age_bin, race + 14 diagnoses per sample
│   └── chexpert_manifest.csv
│
├── chexpert_multibackbone/       # Same as chexpert/ with race_mapping.csv
│   ├── {backbone}_embeddings.npy
│   ├── {backbone}_metadata.csv
│   └── race_mapping.csv
│
├── mimic_cxr/                    # MIMIC-CXR 4 backbones (from qml-mimic-cxr-embeddings)
│   ├── {backbone}_embeddings.npy # 4 backbones × 1491 samples each
│   ├── {backbone}_metadata.csv   # race, sex, age, age_bin + 14 diagnoses per sample
│   ├── mimic_cxr_manifest.csv
│   ├── mimic-cxr-2.0.0-chexpert.csv   # ← FULL dataset (227K studies)
│   └── mimic-cxr-2.0.0-metadata.csv   # ← FULL dataset (377K DICOMs)
│
└── celeba/                       # CelebA (from torchvision, 10k subsample)
    ├── celeba_real_embeddings.npy # (10000, 2048) ResNet-50 ImageNet
    └── celeba_real_metadata.csv   # gender + 40 CelebA attributes
```

## Metadata CSV Format

All metadata CSVs share a common schema:

| Column | Type | Description |
|--------|------|-------------|
| `task_label` | int | Binary task label (0/1) |
| `sex` | str | Male / Female |
| `age` | float | Patient age |
| `age_bin` | str | Age group: <40, 40-60, 60-80, 80+ |
| `race` | str | WHITE, BLACK, ASIAN, OTHER (MIMIC-CXR only) |

**Per-diagnosis columns** (MIMIC-CXR and CheXpert multi-backbone):

| Column | Values | Description |
|--------|--------|-------------|
| `Atelectasis` | 1.0 / 0.0 / NaN | Positive / Negative / Unlabeled |
| `Cardiomegaly` | 1.0 / 0.0 / NaN | |
| `Consolidation` | 1.0 / 0.0 / NaN | |
| `Edema` | 1.0 / 0.0 / NaN | |
| `Enlarged Cardiomediastinum` | 1.0 / 0.0 / NaN | |
| `Fracture` | 1.0 / 0.0 / NaN | |
| `Lung Lesion` | 1.0 / 0.0 / NaN | |
| `Lung Opacity` | 1.0 / 0.0 / NaN | |
| `No Finding` | 1.0 / 0.0 / NaN | |
| `Pleural Effusion` | 1.0 / 0.0 / NaN | |
| `Pleural Other` | 1.0 / 0.0 / NaN | |
| `Pneumonia` | 1.0 / 0.0 / NaN | |
| `Pneumothorax` | 1.0 / 0.0 / NaN | |
| `Support Devices` | 1.0 / 0.0 / NaN | |

## Reproduction Scripts

| Dataset | Extraction Script | Prerequisites |
|---------|-------------------|---------------|
| CheXpert (MedCLIP) | `scripts/setup_chexpert_data.py` | Existing `data/chest_*.npy` |
| CheXpert (multi-backbone) | `scripts/extract_chexpert_hf_multibackbone.py --device mps --parallel` | `pip install datasets`, network access |
| MIMIC-CXR (embeddings) | `scripts/setup_mimic_cxr_data.py` | [qml-mimic-cxr-embeddings](https://huggingface.co/datasets/MITCriticalData/qml-mimic-cxr-embeddings) repo |
| MIMIC-CXR (diagnosis labels) | `scripts/join_mimic_diagnosis_labels.py` | PhysioNet `mimic-cxr-2.0.0-chexpert.csv` |
| CelebA | `scripts/extract_celeba_embeddings.py` | `pip install datasets`, network access |

## Data Provenance

- **CheXpert**: Stanford ML Group. [CheXpert: A Large Chest Radiograph Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/). Images via HuggingFace `danjacobellis/chexpert`.
- **MIMIC-CXR**: Johnson et al. [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/). Embeddings via `MITCriticalData/qml-mimic-cxr-embeddings`. Diagnosis labels from PhysioNet (CheXpert labeler output). Demographics from MIMIC-IV via `subject_id` join.
- **CelebA**: Liu et al. [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Notes

- The `_cache/` subdirectory in `chexpert_multibackbone/` contains raw PIL images cached during extraction. It is excluded from the HuggingFace upload (large binary pickle). Re-run the extraction script to regenerate.
- MIMIC-CXR `*_metadata_orig.csv` files are pre-diagnosis-join backups. The `*_metadata.csv` files contain the joined version with 14 diagnosis columns.
- All random seeds are fixed to 42. CheXpert multi-backbone uses the first 2000 samples from the streaming iterator (deterministic ordering from HuggingFace).
