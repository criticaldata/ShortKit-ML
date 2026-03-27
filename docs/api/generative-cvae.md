# Generative CVAE API

Generative CVAE counterfactual detector for shortcut detection via conditional VAE embedding-space counterfactuals.

## Class Reference

::: shortcut_detect.causal.generative_cvae.src.detector.GenerativeCVEDetector
    options:
      show_root_heading: true
      show_source: true

## Quick Example

```python
from shortcut_detect.causal import GenerativeCVEDetector

detector = GenerativeCVEDetector(epochs=50, random_state=42)
detector.fit(embeddings, group_labels, labels)

results = detector.results_
print(results["shortcut_detected"])
print(results["metrics"])
print(detector.summary())
```

## Unified API Example

```python
from shortcut_detect import ShortcutDetector

detector = ShortcutDetector(methods=["generative_cvae"])
detector.fit(embeddings=emb, labels=labels, group_labels=groups)

print(detector.get_results()["generative_cvae"])
```
