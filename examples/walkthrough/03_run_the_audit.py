"""Step 3. Run the paper's audit on CelebA: does the ResNet-50 embedding carry Male beyond Blond_Hair?

Usage: python examples/walkthrough/03_run_the_audit.py [data_dir] [n_images]
Saves the result to output/walkthrough/celeba_audit.pkl for the report step.
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from shortcut_detect.audit import AuditConfig, conditional_audit

D = sys.argv[1] if len(sys.argv) > 1 else "data/celeba"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

E = np.load(f"{D}/celeba_real_embeddings.npy")
att = pd.read_csv(f"{D}/celeba_real_attributes.csv")
idx = np.sort(np.random.default_rng(0).choice(len(E), N, replace=False))  # a random subset keeps the demo short
E, A, Y = E[idx], att["Male"].to_numpy()[idx], att["Blond_Hair"].to_numpy()[idx]

cfg = AuditConfig(n_permutations=200, n_permutations_probe=500, seed=42)  # the paper's budget
t = time.time()
result = conditional_audit(E, A, Y, config=cfg)  # groups=patient_ids when one unit has several rows
print(f"audit of {N} images x {E.shape[1]} dims in {time.time() - t:.0f} s\n")

probe = result.results["probe"]
print(f"Conditional probe AUROC {probe.statistic:.3f} (95% CI {probe.ci_low:.3f} to {probe.ci_high:.3f})")
print(f"  null mean {probe.null_mean:.3f}, SD {probe.null_sd:.3f}, permutation p = {probe.p_value:.3f} -> flag: {probe.flag}")
print()
print(f"{'detector':16s} {'statistic':>9s} {'null mean':>9s} {'p':>7s} {'q (BH)':>7s}  flag")
for r in result.results.values():
    q = "--" if r.detector == "probe" else f"{r.p_bh:.3f}"
    print(f"{r.detector:16s} {r.statistic:9.3f} {r.null_mean:9.3f} {r.p_value:7.3f} {q:>7s}  {'yes' if r.flag else 'no'}")
print(f"\n{result.n_flagged} of {len(result.results)} detectors flagged; the probe decides, the rest describe.")

os.makedirs("output/walkthrough", exist_ok=True)
with open("output/walkthrough/celeba_audit.pkl", "wb") as f:
    pickle.dump(result, f)
print("saved output/walkthrough/celeba_audit.pkl")
