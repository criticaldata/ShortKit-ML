"""Adversarial debiasing, as shipped in ShortKit-ML (``mitigation.AdversarialDebiasing``).

Learns an encoder whose output predicts the task while an adversary, trained through
a gradient-reversal layer, fails to predict the attribute. A logistic regression on
the encoded embeddings makes the task predictions. Package defaults.
"""

from shortcut_detect.leaderboard.package_methods import EmbeddingTransformClassifier
from shortcut_detect.mitigation import AdversarialDebiasing

METADATA = {
    "name": "shortkit-adversarial-debiasing",
    "title": "Adversarial debiasing (ShortKit-ML)",
    "track": "embedding-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1801.07593", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Gradient-reversal adversary removes the attribute from an encoder; logistic regression on top. Package defaults.",
    "baseline": True,
}


def fit(embeddings, attribute, task_labels, seed=0):
    debias = AdversarialDebiasing(random_state=seed).fit(embeddings, attribute, task_labels)
    return EmbeddingTransformClassifier(debias, seed).fit(embeddings, task_labels)
