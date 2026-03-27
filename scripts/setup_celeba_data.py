#!/usr/bin/env python3
"""Setup CelebA data for the shortcut detection benchmark pipeline.

Attempts to download real CelebA attribute annotations via gdown.  If that
fails (rate limits, network issues), generates synthetic CelebA-like data
with known shortcuts:
  - Blond_Hair strongly correlated with Female (75%)
  - Heavy_Makeup strongly correlated with Female (80%)

Synthetic embeddings (5000 samples, 512-dim) encode group signals in
specific dimensions so shortcuts are detectable:
  - dims 0-9:   gender signal (effect_size=1.0)
  - dims 10-14: Blond_Hair signal

Outputs
-------
data/celeba/celeba_embeddings.npy   (N, 512) float32
data/celeba/celeba_attributes.csv   raw attribute table
data/celeba/celeba_metadata.csv     pipeline-ready metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "celeba"
EMBEDDING_DIM = 512
N_SAMPLES = 5000
SEED = 42


# ---------------------------------------------------------------------------
# Download real CelebA attributes
# ---------------------------------------------------------------------------
def try_download_celeba(output_path: Path) -> bool:
    """Attempt to download list_attr_celeba.txt via gdown.  Returns True on success."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not installed; skipping download attempt.")
        return False

    url = "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Attempting to download CelebA attributes from Google Drive...")
    try:
        gdown.download(url, str(output_path), quiet=False)
        if output_path.exists() and output_path.stat().st_size > 1000:
            print(f"Downloaded to {output_path}")
            return True
        else:
            print("Download produced empty or tiny file; treating as failure.")
            if output_path.exists():
                output_path.unlink()
            return False
    except Exception as exc:
        print(f"Download failed: {exc}")
        if output_path.exists():
            output_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Parse real CelebA attributes
# ---------------------------------------------------------------------------
def parse_real_attrs(attr_path: Path) -> pd.DataFrame:
    """Parse the official list_attr_celeba.txt into a DataFrame."""
    with open(attr_path) as f:
        n_images = int(f.readline().strip())
        header = f.readline().strip().split()
    df = pd.read_csv(
        attr_path,
        sep=r"\s+",
        skiprows=2,
        header=None,
        names=["image_id"] + header,
    )
    for col in header:
        df[col] = (df[col] == 1).astype(int)
    print(f"Parsed {len(df)} rows from real CelebA attributes ({n_images} declared).")
    return df


# ---------------------------------------------------------------------------
# Synthetic CelebA generation
# ---------------------------------------------------------------------------
def generate_synthetic_celeba(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Create synthetic CelebA data with known shortcut correlations.

    Key shortcuts:
      - Blond_Hair: 75% of blonds are female
      - Heavy_Makeup: 80% of makeup-positive are female
    """
    rng = np.random.default_rng(seed)

    # Male ~42% (as in real CelebA)
    male = rng.binomial(1, 0.42, n_samples)
    female = 1 - male

    # --- Known shortcuts (correlated with gender) ---
    # Blond_Hair: ~15% prevalence, 75% of positives are female
    # P(blond|female) ~ 0.20, P(blond|male) ~ 0.07
    blond_p = np.where(female, 0.20, 0.07)
    blond_hair = rng.binomial(1, blond_p)

    # Heavy_Makeup: ~37% prevalence, 80% of positives are female
    # P(makeup|female) ~ 0.55, P(makeup|male) ~ 0.10
    makeup_p = np.where(female, 0.55, 0.10)
    heavy_makeup = rng.binomial(1, makeup_p)

    # Smiling: ~48%, weakly correlated
    smiling_p = np.where(female, 0.52, 0.43)
    smiling = rng.binomial(1, smiling_p)

    # Other attributes (for completeness)
    young = rng.binomial(1, 0.77, n_samples)
    eyeglasses = rng.binomial(1, np.where(male, 0.08, 0.05))
    attractive = rng.binomial(1, np.where(female, 0.65, 0.33))
    wearing_lipstick = rng.binomial(1, np.where(female, 0.75, 0.10))
    no_beard = rng.binomial(1, np.where(female, 0.98, 0.65))
    high_cheekbones = rng.binomial(1, np.where(female, 0.52, 0.36))
    bald = rng.binomial(1, np.where(male, 0.04, 0.001))
    bangs = rng.binomial(1, np.where(female, 0.20, 0.08))
    black_hair = rng.binomial(1, 0.24, n_samples)
    brown_hair = rng.binomial(1, 0.20, n_samples)
    gray_hair = rng.binomial(1, 0.04, n_samples)
    big_nose = rng.binomial(1, np.where(male, 0.28, 0.15))
    goatee = rng.binomial(1, np.where(male, 0.08, 0.001))
    mustache = rng.binomial(1, np.where(male, 0.06, 0.001))
    sideburns = rng.binomial(1, np.where(male, 0.07, 0.001))
    wearing_earrings = rng.binomial(1, np.where(female, 0.25, 0.03))
    wearing_hat = rng.binomial(1, 0.05, n_samples)

    df = pd.DataFrame(
        {
            "image_id": [f"{i+1:06d}.jpg" for i in range(n_samples)],
            "Male": male,
            "Young": young,
            "Blond_Hair": blond_hair,
            "Eyeglasses": eyeglasses,
            "Smiling": smiling,
            "Heavy_Makeup": heavy_makeup,
            "Wearing_Lipstick": wearing_lipstick,
            "Attractive": attractive,
            "Big_Nose": big_nose,
            "Bald": bald,
            "Bangs": bangs,
            "Black_Hair": black_hair,
            "Brown_Hair": brown_hair,
            "Goatee": goatee,
            "Gray_Hair": gray_hair,
            "High_Cheekbones": high_cheekbones,
            "Mustache": mustache,
            "No_Beard": no_beard,
            "Sideburns": sideburns,
            "Wearing_Earrings": wearing_earrings,
            "Wearing_Hat": wearing_hat,
        }
    )

    print(f"Generated synthetic CelebA data: {len(df)} samples, {len(df.columns)-1} attributes.")
    return df


# ---------------------------------------------------------------------------
# Embedding generation (512-dim with structured signal)
# ---------------------------------------------------------------------------
def generate_embeddings(df: pd.DataFrame, dim: int = EMBEDDING_DIM, seed: int = SEED) -> np.ndarray:
    """Create 512-dim synthetic embeddings with structured shortcut signals.

    - dims 0-9:   gender signal (effect_size=1.0)
    - dims 10-14: Blond_Hair signal
    - dims 15+:   Gaussian noise
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32) * 0.3

    # Gender signal in first 10 dims (effect_size=1.0)
    male = df["Male"].values.astype(np.float32)
    gender_direction = rng.standard_normal((10,)).astype(np.float32)
    gender_direction /= np.linalg.norm(gender_direction)
    embeddings[:, :10] += np.outer(male * 1.0 - (1 - male) * 1.0, gender_direction)

    # Blond_Hair signal in dims 10-14
    blond = df["Blond_Hair"].values.astype(np.float32)
    blond_direction = rng.standard_normal((5,)).astype(np.float32)
    blond_direction /= np.linalg.norm(blond_direction)
    embeddings[:, 10:15] += np.outer(blond * 0.8 - (1 - blond) * 0.2, blond_direction)

    # Heavy_Makeup signal in dims 15-19
    makeup = df["Heavy_Makeup"].values.astype(np.float32)
    makeup_direction = rng.standard_normal((5,)).astype(np.float32)
    makeup_direction /= np.linalg.norm(makeup_direction)
    embeddings[:, 15:20] += np.outer(makeup * 0.7 - (1 - makeup) * 0.2, makeup_direction)

    # Smiling signal in dims 20-24 (weaker)
    smiling = df["Smiling"].values.astype(np.float32)
    smile_direction = rng.standard_normal((5,)).astype(np.float32)
    smile_direction /= np.linalg.norm(smile_direction)
    embeddings[:, 20:25] += np.outer(smiling * 0.4 - (1 - smiling) * 0.1, smile_direction)

    print(f"Generated embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}")
    return embeddings


# ---------------------------------------------------------------------------
# Build pipeline-ready metadata
# ---------------------------------------------------------------------------
def build_metadata(
    df: pd.DataFrame, task_label: str = "Smiling", group_label: str = "Male"
) -> pd.DataFrame:
    """Build metadata CSV with image_id, task_label, group_label, and attr_* columns."""
    attr_cols = [c for c in df.columns if c != "image_id"]
    meta = pd.DataFrame(
        {
            "image_id": df["image_id"],
            "task_label": df[task_label].values,
            "group_label": df[group_label].values,
        }
    )
    for col in attr_cols:
        meta[f"attr_{col}"] = df[col].values
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup CelebA data for shortcut detection benchmark"
    )
    parser.add_argument(
        "--real-attrs",
        type=Path,
        default=DATA_DIR / "list_attr_celeba.txt",
        help="Path to real list_attr_celeba.txt",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help="Number of synthetic samples (default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skip-download", action="store_true", help="Skip gdown download attempt")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" CelebA Data Setup")
    print("=" * 60)

    # Try to get real CelebA data
    have_real = args.real_attrs.exists()
    if not have_real and not args.skip_download:
        have_real = try_download_celeba(args.real_attrs)

    if have_real:
        print(f"Using real CelebA attributes from {args.real_attrs}")
        df = parse_real_attrs(args.real_attrs)
        if "image_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "image_id"})
    else:
        print(f"Generating synthetic CelebA data ({args.n_samples} samples)...")
        df = generate_synthetic_celeba(n_samples=args.n_samples, seed=args.seed)

    # Save raw attributes
    attr_csv = args.output_dir / "celeba_attributes.csv"
    df.to_csv(attr_csv, index=False)
    print(f"Saved attributes to {attr_csv}")

    # Generate embeddings
    embeddings = generate_embeddings(df, dim=EMBEDDING_DIM, seed=args.seed)
    emb_path = args.output_dir / "celeba_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings to {emb_path}")

    # Build metadata
    metadata = build_metadata(df, task_label="Smiling", group_label="Male")
    meta_path = args.output_dir / "celeba_metadata.csv"
    metadata.to_csv(meta_path, index=False)
    print(f"Saved metadata to {meta_path}")

    # Summary statistics for known shortcuts
    print("\n--- Known Shortcut Statistics ---")
    female = 1 - df["Male"].values
    for attr in ["Blond_Hair", "Heavy_Makeup", "Wearing_Lipstick"]:
        if attr not in df.columns:
            continue
        pos = df[attr].values == 1
        female_rate = female[pos].mean() if pos.sum() > 0 else 0.0
        print(f"  {attr}: {pos.sum()} positive, {female_rate:.1%} are female")

    print(f"\n  Total samples:   {len(df)}")
    print(f"  Embedding dim:   {EMBEDDING_DIM}")
    print(f"  Male ratio:      {df['Male'].mean():.1%}")
    print("\nCelebA data setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
