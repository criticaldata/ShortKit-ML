"""Per-dimension group tests, as shipped in ShortKit-ML (``ShortcutDetector(methods=["statistical"])``).

Tests every embedding dimension for a difference between attribute groups, with FDR correction. Score: -log10 of the smallest corrected p-value.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-statistical",
    "title": "Per-dimension group tests (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": [
        "https://doi.org/10.1214/aoms/1177730491",
        "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
        "https://doi.org/10.64898/2026.04.29.26352053",
    ],
    "description": "Tests every embedding dimension for a difference between attribute groups, with FDR correction; the package's built-in `statistical` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("statistical", embeddings, attribute, task_labels, seed=seed)
