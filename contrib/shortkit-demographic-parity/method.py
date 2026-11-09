"""Demographic parity gap, as shipped in ShortKit-ML (``ShortcutDetector(methods=["demographic_parity"])``).

Trains a task classifier on the embeddings and compares positive-prediction rates across attribute groups. Score: the parity gap.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-demographic-parity",
    "title": "Demographic parity gap (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1412.3756", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Trains a task classifier on the embeddings and compares positive-prediction rates across attribute groups; the package's built-in `demographic_parity` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("demographic_parity", embeddings, attribute, task_labels, seed=seed)
