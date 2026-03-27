# Detector Base Contract

This page describes the **detector contract** implemented by all standalone detection methods.

## Required Interface

All detectors should subclass `DetectorBase` and provide:

- `fit(...) -> self`: runs the method and populates results
- `get_report() -> dict`: returns a standardized report (`results_`)
- `summary() -> str`: short, human-readable summary

## Required Attributes

After `fit()` completes, detectors must set:

- `results_`: a standardized dictionary (see schema below)
- `shortcut_detected_`: `True` / `False` / `None`
- `_is_fitted`: `True`

## RiskLevel Enum

The `RiskLevel` enum normalizes risk values across all detectors:

```python
from shortcut_detect.detector_base import RiskLevel

class RiskLevel(str, Enum):
    LOW = "low"          # Little or no shortcut evidence
    MODERATE = "moderate" # Some evidence; warrants investigation
    HIGH = "high"        # Strong shortcut evidence
    UNKNOWN = "unknown"  # Cannot determine (insufficient data, not implemented)
```

### Key methods

| Method | Description |
|---|---|
| `RiskLevel.from_string(value)` | Normalize a string to a `RiskLevel` member. Accepts `"low"`, `"moderate"`, `"high"`, `"unknown"`, and the legacy alias `"medium"` (mapped to `MODERATE`). Returns `UNKNOWN` for `None` or unrecognized values. |
| `risk.to_display()` | Returns a capitalized label, e.g. `"High"`. |

```python
>>> RiskLevel.from_string("medium")
<RiskLevel.MODERATE: 'moderate'>

>>> RiskLevel.HIGH.to_display()
'High'
```

## Standard `results_` Schema

```python
{
    "method": str,
    # Unique snake_case identifier set in __init__
    # (e.g. "demographic_parity", "probe", "geometric").

    "shortcut_detected": bool | None,
    # True  -> shortcut evidence found
    # False -> no shortcut evidence
    # None  -> method cannot determine

    "risk_level": "low" | "moderate" | "high" | "unknown",
    # Normalized automatically by RiskLevel.from_string()

    "metrics": {
        # Small, scalar summaries suitable for tables and dashboards.
        # Example: {"accuracy": 0.95, "dp_gap": 0.12}
        # Do NOT place arrays or large objects here.
    },

    "notes": str,
    # Human-readable explanation of the finding.

    "metadata": {
        # Configuration values, dataset counts, or other non-metric context.
        # Example: {"n_samples": 5000, "threshold": 0.1}
    },

    # --- optional keys (only present when not None) ---

    "report": {
        # Detailed structured report: per-group breakdowns, confusion
        # matrices, etc.
    },

    "details": {
        # Large or auxiliary outputs: arrays, plots, model weights.
        # Consumers should not assume these are small.
    },
}
```

### Formatting Guidance

- Keep `metrics` small and scalar (for easy display in summaries/reports).
- Place arrays, tables, and large objects in `report` or `details`.
- Use `risk_level="unknown"` if no clear risk signal is available.
- Use `shortcut_detected=None` if the method is not yet implemented.

## `_set_results()` Reference

This is the **only** recommended way to build `results_`.  It normalizes the risk level and guarantees schema compliance.

```python
def _set_results(
    self,
    *,
    shortcut_detected: bool | None,
    risk_level: str | RiskLevel = RiskLevel.UNKNOWN,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
    metadata: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `shortcut_detected` | `bool \| None` | Whether shortcut evidence was found. `True` = detected, `False` = not detected, `None` = cannot determine. |
| `risk_level` | `str \| RiskLevel` | Assessed risk level. Accepts `"low"`, `"moderate"`, `"high"`, `"unknown"`, or a `RiskLevel` member. Legacy `"medium"` is mapped to `"moderate"`. Default: `RiskLevel.UNKNOWN`. |
| `metrics` | `dict \| None` | Small scalar metrics for dashboards. Example: `{"accuracy": 0.95}`. Default: `{}`. |
| `notes` | `str` | Human-readable explanation. Default: `""`. |
| `metadata` | `dict \| None` | Non-metric context (config, counts). Default: `{}`. |
| `report` | `dict \| None` | Detailed structured report. Only added to `results_` when not `None`. |
| `details` | `dict \| None` | Large/auxiliary outputs. Only added to `results_` when not `None`. |

## `fit()` Contract

Subclasses **must** implement `fit()`.  Inside `fit()`, the implementation should:

1. Accept whatever data the method needs (typically `embeddings`, `labels`, and/or `group_labels`).
2. Validate inputs (shapes, types, constraints).
3. Compute detection metrics.
4. Determine `shortcut_detected` (bool or None).
5. Assess `risk_level`.
6. Call `self._set_results(...)` with the outcome.
7. Set `self.shortcut_detected_` to the boolean outcome.
8. Set `self._is_fitted = True` **after** `_set_results()`.
9. Return `self`.

## Minimal Example

```python
from shortcut_detect.detector_base import DetectorBase, RiskLevel

class MyDetector(DetectorBase):
    def __init__(self, threshold: float = 0.5):
        super().__init__(method="my_detector")
        self.threshold = threshold

    def fit(self, embeddings, labels):
        score = 0.42  # computed by your method
        shortcut = score >= self.threshold
        self.shortcut_detected_ = shortcut

        self._set_results(
            shortcut_detected=shortcut,
            risk_level="high" if shortcut else "low",
            metrics={"score": score},
            notes="Example detector",
            metadata={"threshold": self.threshold},
            report={"score": score},
        )
        self._is_fitted = True
        return self
```

## Full Template

For a complete, copy-paste-ready implementation including both `DetectorBase` and `BaseDetector` patterns, see [`shortcut_detect/detector_template.py`](https://github.com/criticaldata/ShortKit-ML/blob/main/shortcut_detect/detector_template.py).
