"""Occlusion flip consistency: is there one spot that changes many predictions?

Grays out a 4x4 square at every position (stride 2) and records which images change
their predicted class. A model that reads the task from the object depends on where
that object sits, which varies per image, so no single position flips many images. A
model that reads a fixed-position cue flips the same images at the same position.
The score is the largest share of images any single position flips. Needs only
forward passes and no attribute labels or patch location.
"""

from shortcut_detect.leaderboard.saliency import occlusion_flips

METADATA = {
    "name": "occlusion-flip-consistency",
    "title": "Occlusion flip consistency",
    "track": "image-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/1311.2901",
    "description": "Largest share of predictions flipped by graying one 4x4 position; flags above 15%.",
    "baseline": True,
}


def detect(images, task_labels, model, seed=0):
    rates, maps = occlusion_flips(model, images, size=4, stride=2)
    score = float(rates.max())
    return {"score": score, "flag": score > 0.15, "heatmaps": maps}
