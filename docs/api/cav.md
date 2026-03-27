# CAV API

Concept Activation Vector detector for concept-level shortcut testing.

## Class Reference

::: shortcut_detect.xai.cav.src.detector.CAVDetector
    options:
      show_root_heading: true
      show_source: true

## Loader Integration Example

```python
from shortcut_detect import ShortcutDetector

loader_data = {
    "concept_sets": {
        "shortcut_concept": concept_examples,
        "control_concept": control_examples,
    },
    "random_set": random_examples,
    "target_activations": target_activations,
    "target_directional_derivatives": target_directional_derivatives,
}

detector = ShortcutDetector(methods=["cav"])
detector.fit_from_loaders({"cav": loader_data})

print(detector.get_results()["cav"]["metrics"])
```
