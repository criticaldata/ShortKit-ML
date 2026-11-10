"""Attribute probe, as shipped in ShortKit-ML (``ShortcutDetector(methods=["probe"])``).

Trains a classifier to predict the attribute from the embeddings. Score: held-out F1 of that probe.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-probe",
    "title": "Attribute probe (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1610.01644", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Trains a classifier to predict the attribute from the embeddings; the package's built-in `probe` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("probe", embeddings, attribute, task_labels, seed=seed)
