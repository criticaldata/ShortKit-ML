"""Group-balanced subsampling, in the spirit of deep feature reweighting (DFR).

Retrains the linear head on a subsample with equally many examples from each
(task, attribute) group (Kirichenko et al. 2023 use a held-out set; here the
training set is all there is).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

METADATA = {
    "name": "group-balanced-subsample",
    "title": "Group-balanced subsample (DFR-style)",
    "track": "embedding-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/2110.14503",
    "description": "Logistic regression on equal-size random draws from each (task, attribute) group.",
    "baseline": True,
}


def fit(embeddings, attribute, task_labels, seed=0):
    rng = np.random.default_rng(seed)
    g = task_labels * 2 + attribute
    m = np.bincount(g, minlength=4).min()
    idx = np.concatenate([rng.choice(np.flatnonzero(g == k), m, replace=False) for k in range(4)])
    return LogisticRegression(C=1.0, max_iter=2000).fit(embeddings[idx], task_labels[idx])
