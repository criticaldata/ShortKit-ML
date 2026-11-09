"""Shortcut masking, as shipped in ShortKit-ML (``mitigation.ShortcutMasker``), detect-then-mitigate.

The benchmark does not say where the shortcut is, so the method finds it first, from
the group labels (allowed on mitigation tracks): the 4x4 square whose pixels differ most
between attribute groups within each task label. It masks that region in every
training image with the package's ShortcutMasker ("randomize" strategy) and trains the
shared CNN on the masked images.
"""

import numpy as np

from shortcut_detect.leaderboard.image_models import TorchPredictor, train_classifier
from shortcut_detect.leaderboard.package_methods import locate_shortcut
from shortcut_detect.mitigation import ShortcutMasker

METADATA = {
    "name": "shortkit-shortcut-masking",
    "title": "Shortcut masking (ShortKit-ML)",
    "track": "image-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": [
        "https://doi.org/10.1145/3306618.3314293",
        "https://doi.org/10.64898/2026.04.29.26352053",
    ],
    "description": "Locate the region that differs most between attribute groups, randomize it in training images with ShortcutMasker, train.",
    "baseline": True,
}


def fit(images, attribute, task_labels, seed=0):
    mask = locate_shortcut(images, attribute, task_labels)
    masker = ShortcutMasker(strategy="randomize", random_state=seed)
    channels_last = np.transpose(images, (0, 2, 3, 1))
    masked = masker.mask_images(
        channels_last, shortcut_masks=np.broadcast_to(mask, images.shape[:1] + mask.shape)
    )
    masked = np.ascontiguousarray(np.transpose(masked, (0, 3, 1, 2)), dtype=np.float32)
    return TorchPredictor(train_classifier(masked, task_labels, seed=seed))
