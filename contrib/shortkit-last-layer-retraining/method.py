"""Last-layer retraining, as shipped in ShortKit-ML (``mitigation.LastLayerRetraining``).

Builds a group-balanced subset of the training set and fits an L1 logistic
regression on it (the DFR recipe applied to fixed embeddings). Package defaults.
"""

from shortcut_detect.mitigation import LastLayerRetraining

METADATA = {
    "name": "shortkit-last-layer-retraining",
    "title": "Last-layer retraining (ShortKit-ML)",
    "track": "embedding-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/2204.02937", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Group-balanced subset plus L1 logistic regression; the package's LastLayerRetraining with defaults.",
    "baseline": True,
}


def fit(embeddings, attribute, task_labels, seed=0):
    return LastLayerRetraining(random_state=seed).fit(embeddings, task_labels, attribute)
