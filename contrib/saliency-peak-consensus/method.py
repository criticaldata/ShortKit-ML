"""Saliency-peak consensus: do the model's explanations all point at the same spot?

For every image, find the pixel where occluding the input hurts the predicted
class most. If the model reads the task from the object, those peaks
follow the object and scatter across the image; if it reads a fixed-location
shortcut, they pile up in one place. Score = the largest share of images whose
peaks fall inside any one 5x5 window. In the spirit of spectral relevance analysis (SpRAy,
Lapuschkin et al. 2019), which clusters explanations to find Clever Hans
behavior, reduced to its simplest form.
"""

import numpy as np

from shortcut_detect.leaderboard.saliency import occlusion_maps

METADATA = {
    "name": "saliency-peak-consensus",
    "title": "Saliency-peak consensus",
    "track": "image-detection",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://doi.org/10.1038/s41467-019-08987-4",
    "description": "Largest share of images whose occlusion-sensitivity peaks share one 5x5 window.",
    "baseline": True,
}


def detect(images, task_labels, model, seed=0):
    maps = occlusion_maps(model, images)
    n, H, W = maps.shape
    rows, cols = np.unravel_index(maps.reshape(n, -1).argmax(1), (H, W))
    hist = np.zeros((H + 4, W + 4))
    np.add.at(hist, (rows + 2, cols + 2), 1)
    c = hist.cumsum(0).cumsum(1)  # 5x5 window sums via an integral image
    c = np.pad(c, ((1, 0), (1, 0)))
    windows = c[5:, 5:] - c[:-5, 5:] - c[5:, :-5] + c[:-5, :-5]
    return {"score": float(windows.max() / n), "heatmaps": maps}
