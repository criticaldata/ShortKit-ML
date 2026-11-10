"""Right for the Right Reasons, as shipped in ShortKit-ML (``mitigation.ExplanationRegularization``).

Locate-then-mitigate: find the 4x4 square whose pixels differ most between attribute
groups within each task label (group labels are allowed on mitigation tracks), train
the shared CNN, then fine-tune it with the package's RRR penalty on input gradients
inside that region. Package
defaults except batch size 64 (default 8) to keep the run short.
"""

import torch

from shortcut_detect.leaderboard.image_models import TorchPredictor, train_classifier
from shortcut_detect.leaderboard.package_methods import locate_shortcut
from shortcut_detect.mitigation import ExplanationRegularization

METADATA = {
    "name": "shortkit-explanation-regularization",
    "title": "Explanation regularization, RRR (ShortKit-ML)",
    "track": "image-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": ["https://arxiv.org/abs/1703.03717", "https://doi.org/10.64898/2026.04.29.26352053"],
    "description": "Locate the region that differs most between attribute groups, then fine-tune with an input-gradient penalty there.",
    "baseline": True,
}


def fit(images, attribute, task_labels, seed=0):
    mask = locate_shortcut(images, attribute, task_labels)
    model = train_classifier(images, task_labels, seed=seed)
    rrr = ExplanationRegularization(batch_size=64, device="cpu", random_state=seed)
    rrr.fit(model, torch.as_tensor(images), task_labels, mask)
    model.eval()
    return TorchPredictor(model)
