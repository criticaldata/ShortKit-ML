"""Equalized odds gap, as shipped in ShortKit-ML (``ShortcutDetector(methods=["equalized_odds"])``).

Trains a task classifier on the embeddings and compares true- and false-positive rates across attribute groups. Score: the larger gap.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-equalized-odds",
    "title": "Equalized odds gap (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1610.02413", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Trains a task classifier on the embeddings and compares true- and false-positive rates across attribute groups; the package's built-in `equalized_odds` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("equalized_odds", embeddings, attribute, task_labels, seed=seed)
