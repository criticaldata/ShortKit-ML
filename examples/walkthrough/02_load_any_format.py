"""Step 2. Embeddings can come from any file and any encoder; the audit only sees a 2-D float matrix.

Usage: python examples/walkthrough/02_load_any_format.py [data_dir]
Writes small sample files under output/walkthrough/formats/ from the CelebA arrays, then loads each one back.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

D = sys.argv[1] if len(sys.argv) > 1 else "data/celeba"
OUT = "output/walkthrough/formats"
os.makedirs(OUT, exist_ok=True)

E = np.load(f"{D}/celeba_real_embeddings.npy")[:300]
att = pd.read_csv(f"{D}/celeba_real_attributes.csv").iloc[:300]

# the same 300 rows written four ways, as they might arrive from a pipeline
np.save(f"{OUT}/image_resnet50.npy", E)
pd.DataFrame(E).assign(Male=att["Male"].values, Blond_Hair=att["Blond_Hair"].values).to_csv(f"{OUT}/image_resnet50.csv", index=False)
pd.DataFrame(E).to_parquet(f"{OUT}/image_resnet50.parquet")
torch.save(torch.from_numpy(E), f"{OUT}/image_resnet50.pt")

loaders = {
    ".npy (numpy)": lambda: np.load(f"{OUT}/image_resnet50.npy"),
    ".csv (pandas, label columns dropped)": lambda: pd.read_csv(f"{OUT}/image_resnet50.csv").drop(columns=["Male", "Blond_Hair"]).to_numpy(),
    ".parquet (pandas)": lambda: pd.read_parquet(f"{OUT}/image_resnet50.parquet").to_numpy(),
    ".pt (torch tensor)": lambda: torch.load(f"{OUT}/image_resnet50.pt").numpy(),
}
for name, load in loaders.items():
    X = np.asarray(load(), dtype=np.float32)
    print(f"{name:38s} -> E {X.shape}  {X.dtype}")

print()
print("Width does not matter either: a text encoder (384), a tabular model (8) or a CXR encoder (2048)")
for name, d in [("text, MiniLM", 384), ("tabular, 8 features", 8), ("image, RAD-DINO", 768)]:
    X = np.random.default_rng(0).standard_normal((300, d)).astype(np.float32)
    print(f"{name:38s} -> E {X.shape}  ok")
print()
print("Rule: E is (n, d) float with finite values; A and Y are (n,), aligned with E's rows.")
