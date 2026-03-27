import numpy as np

from shortcut_detect.unified import ShortcutDetector
from shortcut_detect.xai import GradCAMMaskOverlapDetector


def test_gradcam_mask_overlap_with_heatmaps():
    heatmaps = np.zeros((2, 4, 4), dtype=np.float32)
    heatmaps[0, :2, :] = 1.0
    heatmaps[1, 2:, :] = 1.0

    masks = np.zeros((2, 4, 4), dtype=np.float32)
    masks[0, :2, :] = 1.0
    masks[1, :2, :] = 1.0

    detector = GradCAMMaskOverlapDetector(threshold=0.5, mask_threshold=0.5)
    detector.fit(heatmaps=heatmaps, masks=masks)
    report = detector.get_report()

    assert report["metrics"]["n_samples"] == 2
    assert report["metrics"]["attention_in_mask_mean"] > 0.0
    assert report["metrics"]["dice_mean"] >= 0.0
    assert report["metrics"]["iou_mean"] >= 0.0


def test_gradcam_mask_overlap_loader_integration():
    heatmaps = np.ones((3, 3, 3), dtype=np.float32)
    masks = np.ones((3, 3, 3), dtype=np.float32)

    def loader():
        return {"heatmaps": heatmaps, "masks": masks}

    detector = ShortcutDetector(methods=["gradcam_mask_overlap"])
    detector.fit_from_loaders({"gradcam_mask_overlap": loader})

    result = detector.get_results().get("gradcam_mask_overlap")
    assert result is not None
    assert result.get("success")
    assert result.get("metrics", {}).get("n_samples") == 3
