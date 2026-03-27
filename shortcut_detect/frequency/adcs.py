import numpy as np


def compute_adcs(embeddings: np.ndarray, labels: np.ndarray):
    """
    Embedding-adapted ADCS.
    Computes class-wise signed dominance of embedding dimensions.

    Returns:
        adcs: dict[class -> np.ndarray(dim,)]
        energy: dict[class -> np.ndarray(dim,)]
    """
    classes = np.unique(labels)
    energy = {}
    for c in classes:
        z = embeddings[labels == c]
        energy[c] = np.mean(np.abs(z), axis=0)

    adcs = {}
    for c in classes:
        score = np.zeros(embeddings.shape[1])
        for c2 in classes:
            if c2 == c:
                continue
            score += np.sign(energy[c] - energy[c2])
        adcs[c] = score

    return adcs, energy
