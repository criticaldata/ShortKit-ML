"""Plain ERM with the benchmark's shared CNN: the image-mitigation reference."""

from shortcut_detect.leaderboard.image_models import TorchPredictor, train_classifier

METADATA = {
    "name": "erm-cnn",
    "title": "ERM (shared CNN)",
    "track": "image-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://doi.org/10.1109/72.788640",
    "description": "The shared small CNN trained with unweighted cross-entropy, ignoring the attribute.",
    "baseline": True,
}


def fit(images, attribute, task_labels, seed=0):
    return TorchPredictor(train_classifier(images, task_labels, seed=seed))
