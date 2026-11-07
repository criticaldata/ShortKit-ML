"""Step 1. The four arrays the audit needs, shown on CelebA ResNet-50 embeddings.

Usage: python examples/walkthrough/01_what_the_audit_needs.py [data_dir]
data_dir holds celeba_real_embeddings.npy and celeba_real_attributes.csv
(scripts/setup_celeba_data.py downloads them).
"""

import sys

import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "data/celeba"

E = np.load(f"{D}/celeba_real_embeddings.npy")  # one row per image, any width
att = pd.read_csv(f"{D}/celeba_real_attributes.csv")  # one row per image, same order
A = att["Male"].to_numpy()  # the sensitive attribute
Y = att["Blond_Hair"].to_numpy()  # the task the classifier would learn

print(f"E  embeddings   {E.shape}  {E.dtype}   <- any width, any encoder, any modality")
print(f"A  attribute    {A.shape}  values {np.unique(A).tolist()}   share A=1: {A.mean():.3f}")
print(f"Y  task label   {Y.shape}  values {np.unique(Y).tolist()}   share Y=1: {Y.mean():.3f}")
print("groups          optional unit id per row (patient, identity), kept whole in folds and permutations")
print()
print(f"P(A=1 | Y=1) = {A[Y == 1].mean():.3f}    P(A=1 | Y=0) = {A[Y == 0].mean():.3f}")
print("A and Y are correlated, so a plain probe for A would partly read Y.")
print("The audit asks the sharper question: does E carry A beyond what Y explains?")
