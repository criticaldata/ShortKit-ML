"""Group reweighting: weight each (task, attribute) group inversely to its size."""

import numpy as np
from sklearn.linear_model import LogisticRegression

METADATA = {
    "name": "group-reweighting",
    "title": "Group reweighting",
    "track": "embedding-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/2110.14503",
    "description": "Logistic regression with sample weights 1 / |group| over (task, attribute) groups.",
    "baseline": True,
}


def fit(embeddings, attribute, task_labels, seed=0):
    g = task_labels * 2 + attribute
    counts = np.bincount(g, minlength=4).astype(float)
    w = 1.0 / counts[g]
    return LogisticRegression(C=1.0, max_iter=2000).fit(
        embeddings, task_labels, sample_weight=w * len(g) / w.sum()
    )
