"""Embedding frequency shortcut, as shipped in ShortKit-ML (``ShortcutDetector(methods=["frequency"])``).

Probes for attribute signal concentrated in a few high-weight dimensions. Score: probe accuracy.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-frequency",
    "title": "Embedding frequency shortcut (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/2307.09829", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Probes for attribute signal concentrated in a few high-weight dimensions; the package's built-in `frequency` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("frequency", embeddings, attribute, task_labels, seed=seed)
