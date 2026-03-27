# Intersectional Analysis API

Detects fairness gaps across intersections of multiple protected attributes (e.g. race × gender).

## Class Reference

::: shortcut_detect.fairness.intersectional.src.detector.IntersectionalDetector
    options:
      show_root_heading: true
      show_source: true

## Usage Example

```python
from shortcut_detect import ShortcutDetector
import numpy as np

detector = ShortcutDetector(methods=["intersectional"])
detector.fit(
    embeddings=embeddings,
    labels=labels,
    group_labels=np.stack([race_labels, gender_labels], axis=1),
)
print(detector.get_results()["intersectional"])
```

## See Also

- [Intersectional Analysis Method Guide](../methods/intersectional-analysis.md)
- [ShortcutDetector API](shortcut-detector.md)
