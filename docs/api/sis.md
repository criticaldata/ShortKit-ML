# SIS API

Sufficient Input Subsets — finds the minimal subset of embedding dimensions that is sufficient to determine the model's prediction, exposing shortcut features.

## Class Reference

::: shortcut_detect.xai.sis.src.detector.SISDetector
    options:
      show_root_heading: true
      show_source: true

## Usage Example

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["sis"])
detector.fit(embeddings=embeddings, labels=labels)
print(detector.get_results()["sis"])
```

## See Also

- [SIS Method Guide](../methods/sis.md)
- [ShortcutDetector API](shortcut-detector.md)
