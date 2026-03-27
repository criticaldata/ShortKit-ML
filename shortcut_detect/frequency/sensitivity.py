import numpy as np
from sklearn.linear_model import LogisticRegression


def rank_embedding_sensitivity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    top_k: float = 0.05,
):
    """
    Rank embedding dimensions by class-conditional sensitivity.
    Sensitivity = drop in class probability when dimension is masked.

    Returns:
        mask: binary np.ndarray(dim,)
        scores: np.ndarray(dim,)
    """
    z = embeddings
    y = (labels == class_id).astype(int)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(z, y)
    base_prob = clf.predict_proba(z)[:, 1].mean()

    scores = np.zeros(z.shape[1])
    for i in range(z.shape[1]):
        z_masked = z.copy()
        z_masked[:, i] = 0.0
        p = clf.predict_proba(z_masked)[:, 1].mean()
        scores[i] = base_prob - p

    k = max(1, int(top_k * z.shape[1]))
    thresh = np.partition(scores, -k)[-k]
    mask = (scores >= thresh).astype(int)

    return mask, scores
