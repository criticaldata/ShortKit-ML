"""Step 4. Why the audit conditions on the task label: a null data set where A follows Y but E carries no A signal.

Usage: python examples/walkthrough/04_calibration_check.py
Three audits on synthetic data (n=600, d=64, P(A=Y)=0.9): null, null with an unconditional probe, planted shortcut.
"""

import warnings

import numpy as np

from shortcut_detect.audit import AuditConfig, conditional_audit

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
n, d = 600, 64
Y = rng.integers(0, 2, n)
A = np.where(rng.random(n) < 0.9, Y, 1 - Y)  # A agrees with Y 90% of the time
E_null = rng.standard_normal((n, d)).astype(np.float32)
E_null[:, :4] += (Y[:, None] - 0.5) * 1.5  # E encodes Y only: no shortcut potential for A
E_shortcut = E_null.copy()
E_shortcut[:, 4:8] += (A[:, None] - 0.5) * 1.0  # E also encodes A beyond Y

K = {"n_permutations": 100, "n_permutations_probe": 200, "seed": 1}


def show(title, E, **extra):
    r = conditional_audit(E, A, Y, config=AuditConfig(**K, **extra))
    p = r.results["probe"]
    print(f"{title:36s} probe {p.statistic:.3f}  null {p.null_mean:.3f}  p = {p.p_value:.3f}  flag {p.flag!s:5s}  secondary {r.n_flagged - int(p.flag)}/5")


print("Setting: A and Y agree 90% of the time; the embedding encodes Y.\n")
show("null data, conditional audit", E_null)
show("null data, unconditional probe", E_null, include_y=False, permutation="global")
show("planted A signal, conditional audit", E_shortcut)
print()
print("The unconditional probe fires on the null because it reads A through Y.")
print("The conditional audit stays quiet there and fires only when E carries A beyond Y.")
