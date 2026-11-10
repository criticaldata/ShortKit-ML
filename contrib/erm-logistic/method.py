"""Plain empirical risk minimization: the reference every mitigation is compared to."""

from sklearn.linear_model import LogisticRegression

METADATA = {
    "name": "erm-logistic",
    "title": "ERM (logistic regression)",
    "track": "embedding-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://doi.org/10.1109/72.788640",
    "description": "Logistic regression on the raw embeddings, ignoring the attribute.",
    "baseline": True,
}


def fit(embeddings, attribute, task_labels, seed=0):
    return LogisticRegression(C=1.0, max_iter=2000).fit(embeddings, task_labels)
