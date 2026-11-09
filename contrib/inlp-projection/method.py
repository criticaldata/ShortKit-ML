"""Iterative nullspace projection (INLP, Ravfogel et al. 2020).

Repeatedly fits a linear attribute classifier and projects the embeddings onto
its nullspace, then trains the task classifier on the projected embeddings.
Exposes ``transform`` so the benchmark also measures residual leakage.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

METADATA = {
    "name": "inlp-projection",
    "title": "INLP (iterative nullspace projection)",
    "track": "embedding-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/2004.07667",
    "description": "Ten rounds of attribute-classifier nullspace projection, then logistic regression.",
    "baseline": True,
}


class INLP:
    def __init__(self, n_iter=10, seed=0):
        self.n_iter, self.seed = n_iter, seed

    def fit(self, E, a, y):
        d = E.shape[1]
        P = np.eye(d)
        for _ in range(self.n_iter):
            w = LinearSVC(C=0.1, dual=False, random_state=self.seed).fit(E @ P, a).coef_.reshape(-1)
            w = w / (np.linalg.norm(w) + 1e-12)
            P = P @ (np.eye(d) - np.outer(w, w))
        self.P = P
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(self.transform(E), y)
        return self

    def transform(self, E):
        return E @ self.P

    def predict(self, E):
        return self.clf.predict(self.transform(E))


def fit(embeddings, attribute, task_labels, seed=0):
    return INLP(seed=seed).fit(embeddings, attribute, task_labels)
