"""Geometric bias analysis, as shipped in ShortKit-ML (``ShortcutDetector(methods=["geometric"])``).

Measures how far apart the attribute groups sit along their mean-difference direction. Score: largest effect size over group pairs.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-geometric",
    "title": "Geometric bias analysis (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/2505.12540", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Measures how far apart the attribute groups sit along their mean-difference direction; the package's built-in `geometric` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("geometric", embeddings, attribute, task_labels, seed=seed)
