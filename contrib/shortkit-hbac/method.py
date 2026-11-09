"""HBAC clustering, as shipped in ShortKit-ML (``ShortcutDetector(methods=["hbac"])``).

Clusters the embeddings hierarchically and looks for clusters dominated by one attribute group. Score: highest cluster purity.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-hbac",
    "title": "HBAC clustering (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": [
        "https://doi.org/10.1016/j.ipm.2021.102519",
        "https://doi.org/10.64898/2026.04.29.26352053",
    ],
    "description": "Clusters the embeddings hierarchically and looks for clusters dominated by one attribute group; the package's built-in `hbac` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("hbac", embeddings, attribute, task_labels, seed=seed)
