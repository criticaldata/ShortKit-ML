"""GCE high-loss samples, as shipped in ShortKit-ML (``ShortcutDetector(methods=["gce"])``).

Trains with generalized cross-entropy, marks high-loss samples, and tests whether they are concentrated in one attribute group. Score: -log10 of that chi-squared p-value.
The flag is the package's own decision rule for this detector.
"""

from shortcut_detect.leaderboard.package_methods import detect_with

METADATA = {
    "name": "shortkit-gce",
    "title": "GCE high-loss samples (ShortKit-ML)",
    "track": "embedding-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1805.07836", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Trains with generalized cross-entropy, marks high-loss samples, and tests whether they are concentrated in one attribute group; the package's built-in `gce` detector.",
    "baseline": True,
}


def detect(embeddings, attribute, task_labels, seed=0):
    return detect_with("gce", embeddings, attribute, task_labels, seed=seed)
