"""GroupDRO gap, as shipped in ShortKit-ML (``ShortcutDetector(methods=["groupdro"])``).

Trains a GroupDRO task classifier over attribute groups. Score: average minus worst-group accuracy.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-groupdro",
    "title": "GroupDRO gap (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1911.08731", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Trains a GroupDRO task classifier over attribute groups; the package's built-in `groupdro` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("groupdro", embeddings, attribute, task_labels, seed=seed)
