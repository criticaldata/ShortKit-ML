# CAV (Concept Activation Vectors)

Concept Activation Vectors (Kim et al., 2018) test whether model representations encode
human-interpretable shortcut concepts.

This implementation focuses on **precomputed activations** and optional **directional
derivatives** (for TCAV scoring), which fits the library's loader-driven XAI workflow.

## What It Detects

- Whether a concept is linearly separable from a random baseline in activation space.
- Whether model sensitivity aligns with that concept direction (TCAV score).

## Required Inputs

- `concept_sets`: `dict[str, np.ndarray]` where each array is `(n, d)`
- `random_set`: `np.ndarray` `(n, d)` or list of arrays

Optional:

- `target_activations`: `np.ndarray` `(m, d)`
- `target_directional_derivatives`: `np.ndarray` `(m, d)`

If directional derivatives are omitted, TCAV risk is reported as `unknown`.

## Unified API Example

```python
from shortcut_detect import ShortcutDetector

bundle = {
    "concept_sets": {
        "hospital_token": concept_arr,  # (n,d)
        "device_artifact": artifact_arr,
    },
    "random_set": random_arr,  # (n,d)
    "target_activations": target_acts,  # optional
    "target_directional_derivatives": target_dd,  # optional
}

detector = ShortcutDetector(
    methods=["cav"],
    cav_quality_threshold=0.7,
    cav_shortcut_threshold=0.6,
)
detector.fit_from_loaders({"cav": bundle})

result = detector.get_results()["cav"]
print(result["metrics"])
print(result["report"]["per_concept"])
```

## Interpretation

- High concept quality (AUC) + high TCAV indicates likely shortcut reliance.
- A concept is flagged when both thresholds are exceeded.
- Risk levels:
  - `high`: very strong flagged TCAV signal
  - `moderate`: flagged concepts, weaker maximum signal
  - `low`: tested with derivatives, no flagged concepts
  - `unknown`: derivatives not provided

## Reference

Kim, Been, et al. "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)." ICML 2018.
