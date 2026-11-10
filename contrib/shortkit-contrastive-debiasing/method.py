"""Contrastive debiasing, as shipped in ShortKit-ML (``mitigation.ContrastiveDebiasing``).

Learns an encoder with a contrastive loss that pulls together samples with the same
task label across attribute groups, plus a task loss. A logistic regression on the
encoded embeddings makes the task predictions. Package defaults.
"""

from shortcut_detect.leaderboard.package_methods import EmbeddingTransformClassifier
from shortcut_detect.mitigation import ContrastiveDebiasing

METADATA = {
    "name": "shortkit-contrastive-debiasing",
    "title": "Contrastive debiasing (ShortKit-ML)",
    "track": "embedding-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/2203.01517", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Contrastive encoder that aligns same-label samples across attribute groups; logistic regression on top. Package defaults.",
    "baseline": True,
}


def fit(embeddings, attribute, task_labels, seed=0):
    debias = ContrastiveDebiasing(random_state=seed).fit(embeddings, task_labels, attribute)
    return EmbeddingTransformClassifier(debias, seed).fit(embeddings, task_labels)
