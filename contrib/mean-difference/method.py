"""Per-dimension mean difference between attribute groups, within task strata.

For each task label, the largest absolute standardized mean difference
(Cohen's d) between attribute groups over all dimensions; the score is its
average over task labels. Blind to signal carried by interactions or variance.
"""

import numpy as np

METADATA = {
    "name": "mean-difference",
    "title": "Within-task mean difference",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://doi.org/10.4324/9780203771587",
    "description": "Max |Cohen's d| between attribute groups over dimensions, averaged over task strata.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    out = []
    for y in np.unique(task_labels):
        E, a = embeddings[task_labels == y], attribute[task_labels == y]
        g1, g0 = E[a == 1], E[a == 0]
        pooled = np.sqrt((g1.var(0, ddof=1) + g0.var(0, ddof=1)) / 2) + 1e-12
        out.append(np.max(np.abs(g1.mean(0) - g0.mean(0)) / pooled))
    return float(np.mean(out))
