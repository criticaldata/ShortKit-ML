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

This folder holds the inputs read by `scripts/revision/common.py`, the loader behind every result in the revised paper. Nothing under `data/` is committed to GitHub except this README, `.gitignore` and `.gitattributes`. The derived files (embeddings, manifests, metadata, `race_mapping.csv`) are hosted on the gated Hugging Face dataset [MITCriticalData/ShortKit-ML-data](https://huggingface.co/datasets/MITCriticalData/ShortKit-ML-data). Access is granted to users who hold the matching credentials (Stanford for CheXpert, PhysioNet for MIMIC-CXR).

```bash
hf download MITCriticalData/ShortKit-ML-data --repo-type dataset --local-dir data/
```

## Files not on the hub

The Stanford files and the CelebA images are never redistributed; download them from the original source and place them at the paths below. The two MIMIC-CXR files in the last row are rebuilt locally.

| File | Source | Used for |
|------|--------|----------|
| `data/train.csv` | CheXpert-v1.0-small `train.csv`, [Stanford research use agreement](https://stanfordmlgroup.github.io/competitions/chexpert/) | sex, age, view, AP/PA and the 14 labels, joined to the embeddings by image path through `race_mapping.csv` |
| `data/CHEXPERT DEMO.xlsx` | CheXpert demographics file from Stanford | `PRIMARY_RACE` per patient |
| `data/celeba_real/celeba/` | CelebA via `torchvision.datasets.CelebA` (`identity_CelebA.txt`, `list_attr_celeba.txt`, `img_align_celeba/`) | celebrity identity for grouping; images for `scripts/extract_celeba_embeddings.py` |
| `mimic_cxr/vit32_cls_embeddings_realigned.npy`, `mimic_cxr/mimic_race_raw.csv` | Rebuilt by `scripts/revision/prepare_mimic_source.py` from the coreset parquet files in `mimic_cxr/source/` | the realigned ViT-B/32 array and the raw MIMIC race strings |

## Layout

```
data/
├── train.csv                        # from Stanford (see above)
├── CHEXPERT DEMO.xlsx               # from Stanford (see above)
│
├── chexpert_multibackbone/          # CheXpert, 2,000 images, 477 patients
│   ├── chexpert_manifest.csv        # row order, task_label (Pneumothorax), age_bin
│   ├── race_mapping.csv             # image_idx, path, patient_id, PRIMARY_RACE
│   ├── resnet50_embeddings.npy      # (2000, 2048)
│   ├── densenet121_embeddings.npy   # (2000, 1024)
│   ├── vit_b_16_embeddings.npy      # (2000, 768)
│   ├── {backbone}_metadata.csv      # one per backbone, same row order
│   └── public/                      # re-extracted from pinned public checkpoints
│       ├── vit_b_32__vit_b_32__class.npy                      # torchvision ViT_B_32_Weights.DEFAULT
│       ├── dinov2_base__facebook_dinov2-base__pooler_output.npy
│       ├── rad_dino__microsoft_rad-dino__pooler_output.npy
│       └── medsiglip__google_medsiglip-448__vision.npy        # (2000, 1152)
│
├── mimic_cxr/                       # MIMIC-CXR, 1,491 subjects, one image each
│   ├── rad_dino_embeddings.npy      # (1491, 768)
│   ├── vit16_cls_embeddings.npy     # (1491, 768)
│   ├── vit32_cls_embeddings_realigned.npy   # (1491, 768), built by prepare_mimic_source.py
│   ├── medsiglip_embeddings.npy     # (1491, 1152), google/siglip-so400m-patch14-384
│   ├── {backbone}_metadata.csv      # image_path, task_label, race, sex, age, age_bin
│   ├── mimic_race_raw.csv           # image_path, race_raw (built by prepare_mimic_source.py)
│   ├── mimic_cxr_manifest.csv
│   ├── mimic-cxr-2.0.0-chexpert.csv     # PhysioNet label file
│   └── mimic-cxr-2.0.0-metadata.csv     # PhysioNet DICOM metadata
│
├── celeba/                          # CelebA, 10,000 images
│   ├── celeba_real_embeddings.npy   # (10000, 2048), ResNet-50 ImageNet avgpool
│   ├── celeba_real_attributes.csv   # image_id plus the 40 CelebA attributes
│   └── celeba_real_metadata.csv
│
└── celeba_real/celeba/              # torchvision CelebA download (see above)
```

## CheXpert (`chexpert_multibackbone/`)

The 2,000 images are the first 2,000 rows of the official `train.csv`, from 477 patients. Seven backbones are audited. ResNet-50, DenseNet-121 and ViT-B/16 are read from `<backbone>_embeddings.npy`. ViT-B/32, DINOv2, RAD-DINO and MedSigLIP-448 are read from `public/`, which was extracted from the same images with `scripts/revision/extract_missing_chexpert_backbones.py` and pinned public checkpoints; the same code reproduces the stored ViT-B/16 array to cosine 1.000. The older `vit_b_32`, `dinov2_base`, `rad_dino` and `medsiglip` arrays at the top level have no extraction code and are kept only to replay the first submission (`stored=True` in the loader).

Sex, age, view and the labels come from `train.csv`, joined by image path through `race_mapping.csv`, which also gives `patient_id`. The `sex` column in the shipped `_metadata.csv` files is inverted and is not used. Race comes from `CHEXPERT DEMO.xlsx` (`PRIMARY_RACE`, joined by patient): images without a demographics record, or recorded as Unknown or Patient Refused, are missing; Native Hawaiian or Pacific Islander is its own group. The primary analysis compares White against non-White. Cross-validation folds and the permutation null are grouped by patient.

## MIMIC-CXR (`mimic_cxr/`)

1,491 subjects with one image each: the coreset in `mimic_cxr/source/` (strategy 5, PathologyStratifiedClean, 1,999 images, seed 0) minus rows without an in-hospital mortality label. Four backbones: RAD-DINO, ViT-B/16, ViT-B/32 and the array labelled MedSigLIP, which was extracted with `google/siglip-so400m-patch14-384` and is reported in the paper as SigLIP (So400m). The stored `vit32_cls_embeddings.npy` does not match its metadata; the loader reads `vit32_cls_embeddings_realigned.npy` instead. Race is coded from `mimic_race_raw.csv`, with Unknown, Unable to Obtain and Patient Declined treated as missing.

## CelebA (`celeba/`)

10,000 images embedded with an ImageNet ResNet-50 by `scripts/extract_celeba_embeddings.py`, grouped by celebrity identity from `celeba_real/celeba/identity_CelebA.txt`. The attribute is Male; the tasks are Blond_Hair, Heavy_Makeup and Attractive.

## First-submission files

`chest_embeddings.npy`, `chest_labels.npy`, `chest_group_labels.npy`, `chexpert_manifest.csv` and the `chexpert/` folder on the hub belong to the first submission (MedCLIP and the stored backbones) and are only needed to replay it with `scripts/run_dataset2_benchmark.py`.

## Provenance

- **CheXpert**: Stanford ML Group, [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). Images from the `danjacobellis/chexpert` mirror; labels, sex and age from the official `train.csv`.
- **MIMIC-CXR**: Johnson et al., [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) embeddings and metadata in `mimic_cxr/`, with the source parquet files in `mimic_cxr/source/`. Diagnosis labels from PhysioNet; demographics from MIMIC-IV joined by `subject_id`.
- **CelebA**: Liu et al., [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
