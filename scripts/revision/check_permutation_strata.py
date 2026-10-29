"""Patient-level permutation strata (all-negative, all-positive, mixed) per CheXpert task,
on the labelled-only subsets and in every mitigation test half; prints the smallest stratum.
Source of the stratum counts quoted in Methods 3.2. Read-only; writes nothing.
"""
import sys, collections
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import chexpert_meta
from run_head_experiment import attribute_vector, TASKS
from sklearn.model_selection import StratifiedGroupKFold
meta = chexpert_meta(); pid = meta["patient_id"].to_numpy()
def strata(y, p):
    c = collections.Counter()
    for g in np.unique(p):
        ys = np.unique(y[p == g])
        c["allneg" if (len(ys) == 1 and ys[0] == 0) else "allpos" if (len(ys) == 1 and ys[0] == 1) else "mixed"] += 1
    return dict(c)
gmin = 10**9
for t, col in TASKS.items():
    lab = meta[col].to_numpy(dtype=float); keep = (lab == 0) | (lab == 1)
    y = lab[keep].astype(int); p = pid[keep]
    for a in ("sex", "age"):
        ak = attribute_vector(meta, a)[keep]; key = ak * 2 + y
        for s in range(3):
            sgk = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=s)
            tr, te = next(sgk.split(np.zeros((len(y), 1)), key, p))
            st = strata(y[te], p[te]); m = min(st.values()); gmin = min(gmin, m)
            print(t, a, s, st, "test patients", len(set(p[te])))
print("GLOBAL MIN stratum patients in test halves:", gmin)
