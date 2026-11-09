"""Input-gradient hotspot: does the model's saliency pile up in one place?

Computes |d logit_pred / d input| per image (summed over channels), averages the
normalized maps over the dataset and scores how concentrated that average map
is (share of mass in the top 2% of pixels). A shortcut at a fixed location
makes every image's saliency point at the same spot. The per-image maps are
returned as heatmaps.
"""

from shortcut_detect.leaderboard.saliency import input_gradient, top_mass_concentration

METADATA = {
    "name": "input-gradient-hotspot",
    "title": "Input-gradient hotspot",
    "track": "image-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/1312.6034",
    "description": "Concentration of the dataset-averaged vanilla-gradient saliency map.",
    "baseline": True,
}


def detect(images, task_labels, model, seed=0):
    maps = input_gradient(model, images)
    return {"score": top_mass_concentration(maps), "heatmaps": maps}
