"""Occlusion hotspot: where does hiding a small square change the prediction most?

Slides a 4x4 gray square over a 32x32 grid (stride 2), records the drop in the
predicted-class probability, and scores how concentrated the dataset-averaged
drop map is. Model-agnostic: needs only forward passes.
"""

from shortcut_detect.leaderboard.saliency import occlusion_maps, top_mass_concentration

METADATA = {
    "name": "occlusion-hotspot",
    "title": "Occlusion hotspot",
    "track": "image-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/1311.2901",
    "description": "Concentration of the dataset-averaged occlusion-sensitivity map (4x4 square, stride 2).",
    "baseline": True,
}


def detect(images, task_labels, model, seed=0):
    maps = occlusion_maps(model, images, size=4, stride=2)
    return {"score": top_mass_concentration(maps), "heatmaps": maps}
