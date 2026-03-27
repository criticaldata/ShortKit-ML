# Causal Effect API

Causal effect regularization detector for shortcut detection via causal effect estimation.

## Class Reference

::: shortcut_detect.causal.causal_effect.src.detector.CausalEffectDetector
    options:
      show_root_heading: true
      show_source: true

## Loader Integration Example

```python
from shortcut_detect import ShortcutDetector

loader_data = {
    "embeddings": embeddings,   # (n, d)
    "labels": labels,          # (n,)
    "attributes": {
        "race": race_labels,   # (n,) binary or categorical
        "color": color_labels,
    },
}

detector = ShortcutDetector(
    methods=["causal_effect"],
    causal_effect_spurious_threshold=0.1,
)
detector.fit_from_loaders({"causal_effect": loader_data})

print(detector.get_results()["causal_effect"]["metrics"])
```
