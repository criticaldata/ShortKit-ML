#!/usr/bin/env python
"""GradCAM attention analysis on CelebA for paper supplement.

Trains a linear probe on top of ResNet-50 (pretrained, torchvision) using the
pre-extracted CelebA embeddings for the Male attribute.  Then wraps the full
ResNet-50 + probe as a differentiable model and runs GradCAM on target_layer
'layer4' for 16 sample images (8 Male, 8 Female).

Computes face_attention_ratio = sum of GradCAM attention inside face bounding
box divided by total attention, as a proxy for whether the model focuses on
face structure when predicting Male.

Outputs
-------
output/paper_figures/gradcam_celeba_supplement.pdf
    4×4 grid: original image (left) / GradCAM overlay (right) for 16 samples.
output/paper_figures/gradcam_celeba_supplement_metrics.csv
    Per-image metrics: image_id, male_label, face_attention_ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shortcut_detect.gradcam import GradCAMHeatmapGenerator  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
META_PATH = PROJECT_ROOT / "data" / "celeba" / "celeba_real_metadata.csv"
EMB_PATH = PROJECT_ROOT / "data" / "celeba" / "celeba_real_embeddings.npy"
BBOX_PATH = PROJECT_ROOT / "data" / "celeba_real" / "celeba" / "list_bbox_celeba.txt"
IMG_DIR = PROJECT_ROOT / "data" / "celeba_real" / "celeba" / "img_align_celeba"
OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

IMG_SIZE = 224  # ResNet-50 input size
N_SAMPLES = 16  # 8 Male + 8 Female
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation (ResNet-50 pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Combined ResNet-50 + linear probe model
# ---------------------------------------------------------------------------
class ResNetProbe(nn.Module):
    """ResNet-50 backbone with a single linear classification head.

    The backbone's forward produces 2048-d pooled features; the head maps
    those to 2-class logits.  The whole pipeline is differentiable so
    GradCAM can back-propagate through it.
    """

    def __init__(self, probe_weights: torch.Tensor, probe_bias: torch.Tensor) -> None:
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        # Drop the original classifier; keep everything up to avgpool
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = backbone.avgpool  # AdaptiveAvgPool2d(1,1)
        # Linear probe: 2048 -> 2 (Male=1, Female=0)
        self.head = nn.Linear(2048, 2, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(probe_weights)
            self.head.bias.copy_(probe_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)  # (B, 2048, 7, 7) for 224×224
        pooled = self.pool(feat_map).flatten(1)  # (B, 2048)
        return self.head(pooled)  # (B, 2)


# ---------------------------------------------------------------------------
# Train linear probe on pre-extracted embeddings
# ---------------------------------------------------------------------------
def train_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train a 2-class logistic probe on (N, 2048) embeddings."""
    torch.manual_seed(seed)

    X = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    in_dim = X.shape[1]
    probe = nn.Linear(in_dim, 2, bias=True).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = probe(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"    Epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}  acc={acc:.3f}")

    probe.eval()
    with torch.no_grad():
        acc = (probe(X).argmax(1) == y).float().mean().item()
    print(f"  Probe train accuracy: {acc:.3f}")

    return probe.weight.detach().cpu(), probe.bias.detach().cpu()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def load_image_tensor(img_path: Path) -> torch.Tensor:
    """Load single image → (1, 3, 224, 224) normalised tensor."""
    transform = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def load_image_display(img_path: Path) -> np.ndarray:
    """Load single image → (224, 224, 3) uint8 array for display."""
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)


# ---------------------------------------------------------------------------
# Face-attention ratio
# ---------------------------------------------------------------------------
def face_attention_ratio(
    heatmap: np.ndarray,
    bbox: tuple[int, int, int, int],
    img_hw: int = IMG_SIZE,
) -> float:
    """Fraction of GradCAM attention inside the face bounding box.

    Parameters
    ----------
    heatmap : (H, W) array already normalised to [0, 1].
    bbox : (x1, y1, w, h) in original CelebA pixel coords (178×218).
    img_hw : target resolution (heatmap is already this size after GradCAM).
    """
    orig_w, orig_h = 178, 218  # CelebA img_align_celeba fixed size
    x1, y1, bw, bh = bbox
    # Scale bbox to img_hw
    sx = img_hw / orig_w
    sy = img_hw / orig_h
    r1 = max(0, int(y1 * sy))
    r2 = min(img_hw, int((y1 + bh) * sy))
    c1 = max(0, int(x1 * sx))
    c2 = min(img_hw, int((x1 + bw) * sx))

    total = heatmap.sum() + 1e-8
    inside = heatmap[r1:r2, c1:c2].sum()
    return float(inside / total)


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------
def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a RdYlGn heatmap over the original image."""
    cmap = plt.get_cmap("jet")
    hm_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * hm_colored + (1 - alpha) * img_rgb).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("GradCAM Supplement: CelebA Male-attribute attention")
    print("=" * 60)

    # --- Load metadata ---
    print("\n[1/5] Loading metadata and embeddings...")
    meta = pd.read_csv(META_PATH)
    embeddings = np.load(EMB_PATH)
    assert len(meta) == len(embeddings), "Metadata / embedding length mismatch"
    print(f"  {len(meta)} samples, embedding dim {embeddings.shape[1]}")

    # --- Load bboxes (line 0 = total count, line 1 = header) ---
    bbox_df = pd.read_csv(BBOX_PATH, sep=r"\s+", skiprows=1, header=0, low_memory=False)
    bbox_df.columns = ["image_id", "x1", "y1", "w", "h"]
    bbox_dict = {
        row.image_id: (int(row.x1), int(row.y1), int(row.w), int(row.h))
        for row in bbox_df.itertuples()
    }

    # --- Select 8 Male + 8 Female samples whose images exist ---
    print("\n[2/5] Selecting sample images...")
    male_rows = meta[meta["Male"] == 1]
    female_rows = meta[meta["Male"] == 0]

    def pick_existing(rows: pd.DataFrame, n: int) -> pd.DataFrame:
        chosen = []
        for _, row in rows.iterrows():
            path = IMG_DIR / row["image_id"]
            if path.exists():
                chosen.append(row)
            if len(chosen) >= n:
                break
        return pd.DataFrame(chosen)

    male_sel = pick_existing(male_rows, N_SAMPLES // 2)
    female_sel = pick_existing(female_rows, N_SAMPLES // 2)
    sample_df = pd.concat([female_sel, male_sel], ignore_index=True)
    print(f"  Selected {len(female_sel)} Female + {len(male_sel)} Male samples")

    # --- Train probe ---
    print("\n[3/5] Training linear probe on pre-extracted embeddings...")
    labels_all = meta["Male"].values.astype(np.int64)
    w, b = train_probe(embeddings, labels_all)

    # --- Build ResNet-50 + probe model ---
    print("\n[4/5] Building ResNet-50 + probe model and running GradCAM...")
    model = ResNetProbe(probe_weights=w, probe_bias=b).to(DEVICE).eval()
    cam_gen = GradCAMHeatmapGenerator(model, target_layer="features.7")  # layer4 = features[7]

    # --- Run GradCAM ---
    heatmaps = []
    display_imgs = []
    ratios = []
    pred_labels = []

    for _, row in sample_df.iterrows():
        img_path = IMG_DIR / row["image_id"]
        inp = load_image_tensor(img_path).to(DEVICE)
        hm = cam_gen.generate_heatmap(inp, head="logits", target_index=1)  # target: Male=1
        hm_2d = hm[0]  # (224, 224)

        # Face attention ratio
        bbox = bbox_dict.get(row["image_id"], (0, 0, 178, 218))
        ratio = face_attention_ratio(hm_2d, bbox)
        ratios.append(ratio)
        heatmaps.append(hm_2d)

        display_imgs.append(load_image_display(img_path))

        # Model prediction
        with torch.no_grad():
            logits = model(inp)
            pred = logits.argmax(1).item()
        pred_labels.append(pred)

        gt = "M" if row["Male"] == 1 else "F"
        pr = "M" if pred == 1 else "F"
        print(f"  {row['image_id']}  GT={gt}  Pred={pr}  face_ratio={ratio:.3f}")

    cam_gen.close()

    # --- Save metrics CSV ---
    metrics_df = pd.DataFrame(
        {
            "image_id": sample_df["image_id"].values,
            "male_label": sample_df["Male"].values,
            "face_attention_ratio": ratios,
        }
    )
    metrics_path = OUTPUT_DIR / "gradcam_celeba_supplement_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Metrics saved: {metrics_path}")

    mean_ratio = np.mean(ratios)
    male_ratio = np.mean(
        [r for r, row in zip(ratios, sample_df.itertuples(), strict=False) if row.Male == 1]
    )
    female_ratio = np.mean(
        [r for r, row in zip(ratios, sample_df.itertuples(), strict=False) if row.Male == 0]
    )
    print(f"  Mean face_attention_ratio: {mean_ratio:.3f}")
    print(f"    Male: {male_ratio:.3f}  |  Female: {female_ratio:.3f}")

    # --- Generate 4×4 figure ---
    print("\n[5/5] Generating figure...")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, (img, hm, row_data) in enumerate(
        zip(display_imgs, heatmaps, sample_df.itertuples(), strict=False)
    ):
        row_idx = i // 4
        col_idx = i % 4
        ax = axes[row_idx][col_idx]

        overlay = overlay_heatmap(img, hm, alpha=0.45)
        ax.imshow(overlay)
        gt_str = "Male" if row_data.Male == 1 else "Female"
        color = "#0855A5" if row_data.Male == 1 else "#D6264D"
        ax.set_title(
            f"{gt_str}\nFAR={ratios[i]:.2f}",
            fontsize=7,
            color=color,
            pad=2,
        )
        ax.axis("off")

    fig.suptitle(
        "GradCAM attention on CelebA (Male attribute, target=Male)\n"
        f"Mean face-attention ratio: {mean_ratio:.3f}  "
        f"(Male {male_ratio:.3f} | Female {female_ratio:.3f})",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / "gradcam_celeba_supplement.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")

    print("\nDone.")
    print("\n  *** Summary for paper ***")
    print(f"  Mean face-attention ratio = {mean_ratio:.3f}")
    print(f"  Male  = {male_ratio:.3f}  |  Female = {female_ratio:.3f}")


if __name__ == "__main__":
    main()
