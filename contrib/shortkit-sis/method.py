"""Sufficient input subsets, as shipped in ShortKit-ML (``ShortcutDetector(methods=["sis"])``).

Finds the smallest sets of dimensions that alone predict the attribute. Score: minus the mean subset size (smaller sets = more concentrated signal).
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-sis",
    "title": "Sufficient input subsets (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1810.03805", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Finds the smallest sets of dimensions that alone predict the attribute; the package's built-in `sis` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("sis", embeddings, attribute, task_labels, seed=seed)
