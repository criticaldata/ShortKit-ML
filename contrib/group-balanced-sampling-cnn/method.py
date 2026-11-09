"""Group-balanced sampling with the shared CNN.

Each epoch draws examples with replacement with probability 1 / |group| over
(task, attribute) groups, so every group is seen equally often. (Loss
reweighting by 1 / |group| with the same budget was unstable at rho = 0.97,
where minority examples get about 16x weight.)
"""

import numpy as np

from shortcut_detect.leaderboard.image_models import TorchPredictor, train_classifier

METADATA = {
    "name": "group-balanced-sampling-cnn",
    "title": "Group-balanced sampling (shared CNN)",
    "track": "image-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/2110.14503",
    "description": "The shared CNN trained on batches drawn with probability 1 / |group| over (task, attribute) groups.",
    "baseline": True,
}


def fit(images, attribute, task_labels, seed=0):
    g = task_labels * 2 + attribute
    w = 1.0 / np.bincount(g, minlength=4)[g]
    return TorchPredictor(train_classifier(images, task_labels, seed=seed, sampling_weight=w))
