"""Bias direction (PCA), as shipped in ShortKit-ML (``ShortcutDetector(methods=["bias_direction_pca"])``).

Finds the principal direction separating the attribute groups. Score: projection gap between group means on it.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-bias-direction-pca",
    "title": "Bias direction (PCA) (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1607.06520", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Finds the principal direction separating the attribute groups; the package's built-in `bias_direction_pca` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("bias_direction_pca", embeddings, attribute, task_labels, seed=seed)
