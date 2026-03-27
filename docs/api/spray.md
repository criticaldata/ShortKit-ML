# SpRAy API

SpRAy performs spectral clustering of explanation heatmaps to reveal systematic
attention artifacts (Clever Hans behavior).

## Class Reference

::: shortcut_detect.xai.spray_detector.SpRAyDetector
    options:
      show_root_heading: true
      show_source: true

## Usage Example

```python
import numpy as np
from shortcut_detect import SpRAyDetector

heatmaps = np.load("heatmaps.npy")
detector = SpRAyDetector(affinity="cosine", cluster_selection="auto")
detector.fit(heatmaps=heatmaps)

report = detector.get_report()
print(report["report"]["clever_hans"])
```
