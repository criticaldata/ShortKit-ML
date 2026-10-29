"""Patient-level permutation strata and block-permutation partners on CheXpert.

Prints, for the full 2,000-image cohort with Pneumothorax as the task: the three strata of the
stratified permutation (all-negative, all-positive, mixed patients) and, for the block permutation
of the planted-marker audit (patients exchanged only with the same sorted label vector), how many
patients have at least one partner. Then, for the labelled-only subsets of the three head-experiment
tasks: the strata, and the strata in each mitigation test half (StratifiedGroupKFold, seeds 0 to 2)
with the smallest stratum over all of them. Source of the counts quoted in Methods 3.2. Read-only.
"""
import sys, collections
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import chexpert_meta
from sklearn.model_selection import StratifiedGroupKFold

meta = chexpert_meta()
pid = meta["patient_id"].to_numpy()
print("images", len(meta), "patients", len(set(pid)))

def strata(y, p):
    c = collections.Counter()
    for g in np.unique(p):
        ys = np.unique(y[p == g])
        c["allneg" if (len(ys) == 1 and ys[0] == 0) else "allpos" if (len(ys) == 1 and ys[0] == 1) else "mixed"] += 1
    return dict(c)

# full-cohort audit coding: uncertain and blank -> 0
lab = meta["Pneumothorax"].to_numpy(dtype=float)
y_full = (lab == 1).astype(int)
print("Pneumothorax full cohort strata:", strata(y_full, pid))

# block permutation partners for the marker (sorted Y vector key), full cohort
buckets = collections.Counter()
for g in np.unique(pid):
    ys = tuple(sorted(y_full[pid == g].tolist()))
    buckets[ys] += 1
single = sum(1 for k, v in buckets.items() if v == 1)
print("marker block buckets:", len(buckets), "singleton patients:", single,
      "patients with partner:", sum(v for v in buckets.values() if v > 1))
print("singleton keys (n_images, n_pos):", sorted((len(k), sum(k)) for k, v in buckets.items() if v == 1))

# labelled-only subsets per task
TASKS = {"pneumothorax": "Pneumothorax", "pleural_effusion": "Pleural Effusion", "cardiomegaly": "Cardiomegaly"}
def attr(meta, a):
    if a == "sex":
        return (meta["sex"].to_numpy() == "Male").astype(int)
    return (meta["age"].to_numpy(dtype=float) >= 60).astype(int)

for t, col in TASKS.items():
    lab = meta[col].to_numpy(dtype=float)
    keep = (lab == 0) | (lab == 1)
    y = lab[keep].astype(int); p = pid[keep]
    print(t, "labelled-only n=", keep.sum(), "patients", len(set(p)), "strata", strata(y, p))
    for a in ("sex", "age"):
        ak = attr(meta, a)[keep]
        key = ak * 2 + y
        mins = []
        for s in range(3):
            sgk = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=s)
            tr, te = next(sgk.split(np.zeros((len(y), 1)), key, p))
            st = strata(y[te], p[te])
            mins.append(st)
        print("   ", a, "test-half strata seeds 0..2:", mins)

# smallest stratum over every mitigation test half (all tasks, attributes, seeds)
import itertools
allmin = []
for t, col in TASKS.items():
    lab = meta[col].to_numpy(dtype=float)
    keep = (lab == 0) | (lab == 1)
    y = lab[keep].astype(int); p = pid[keep]
    for a in ("sex", "age"):
        key = attr(meta, a)[keep] * 2 + y
        for s in range(3):
            tr, te = next(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=s).split(np.zeros((len(y), 1)), key, p))
            allmin.append(min(strata(y[te], p[te]).values()))
print("smallest stratum in any mitigation test half:", min(allmin))
