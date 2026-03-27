"""
ShortKit-ML MCP Server

Exposes ShortKit-ML shortcut detection capabilities as MCP tools so that
AI assistants (Claude, Cursor, etc.) can run analyses directly from chat.

Usage:
    uv run python -m shortcut_detect.mcp_server
    # or via entry point:
    shortkit-ml-mcp
"""

from __future__ import annotations

import json
import traceback
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ShortKit-ML")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAST_RESULTS: dict[str, Any] = {}  # in-process cache keyed by session_id


def _parse_array(data: list | None, name: str) -> np.ndarray | None:
    """Convert a JSON-serialisable list (or list-of-lists) to numpy array."""
    if data is None:
        return None
    try:
        return np.array(data)
    except Exception as exc:
        raise ValueError(f"Could not parse '{name}' as numeric array: {exc}") from exc


def _safe_serialize(obj: Any) -> Any:
    """Recursively convert numpy/non-serialisable objects to plain Python."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Skip non-serialisable objects (detector instances, etc.)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# Tool: list_methods
# ---------------------------------------------------------------------------


@mcp.tool()
def list_methods() -> str:
    """List all available shortcut detection methods with short descriptions.

    Returns a JSON object mapping method name → description.
    No input required.
    """
    methods = {
        "hbac": "Hierarchical Bias-Aware Clustering — finds demographic clusters with unequal error rates.",
        "probe": "Linear probe — trains a classifier to predict group labels from embeddings.",
        "statistical": "Statistical tests (KS, Wasserstein) on per-feature group distributions.",
        "geometric": "Geometric analysis of embedding space separation between groups.",
        "bias_direction_pca": "PCA-based bias direction — projects embeddings along the most-biased axis.",
        "equalized_odds": "Fairness: checks equal true/false positive rates across groups.",
        "demographic_parity": "Fairness: checks equal positive prediction rates across groups.",
        "intersectional": "Intersectional fairness across all combinations of protected attributes.",
        "frequency": "Detects shortcuts concentrated in a few embedding dimensions (frequency domain).",
        "ssa": "Sufficiency Score Analysis — measures how much protected attributes explain predictions.",
        "vae": "VAE latent space analysis — flags entangled shortcut dimensions.",
        "cav": "Concept Activation Vectors — tests whether a concept is linearly encoded.",
        "sis": "Sufficient Input Subsets — finds minimal feature subsets that determine predictions.",
        "gradcam_mask_overlap": "GradCAM mask overlap — measures attention leakage onto protected regions.",
        "causal_effect": "Causal effect estimation of protected attribute on model output.",
        "generative_cvae": "Generative CVAE counterfactuals — generates counterfactual embeddings to isolate shortcuts.",
        "gce": "Group Conditional Embeddings — measures inter-group embedding divergence.",
        "groupdro": "Group DRO alignment — trains a group-robust model and measures shortcut reduction.",
        "early_epoch_clustering": "Early-epoch clustering — detects shortcuts that emerge in the first training epochs.",
    }
    return json.dumps(methods, indent=2)


# ---------------------------------------------------------------------------
# Tool: generate_synthetic_data
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_synthetic_data(
    n_samples: int = 200,
    n_features: int = 32,
    shortcut_strength: float = 0.8,
    shortcut_type: str = "linear",
    n_classes: int = 2,
    seed: int = 42,
) -> str:
    """Generate a synthetic shortcut dataset for testing detectors.

    Args:
        n_samples: Number of samples (default 200).
        n_features: Embedding dimensionality (default 32).
        shortcut_strength: How strongly the protected attribute correlates with
            the label (0 = none, 1 = perfect). Default 0.8.
        shortcut_type: "linear" or "nonlinear" or "none". Default "linear".
        n_classes: Number of label classes (default 2).
        seed: Random seed (default 42).

    Returns:
        JSON with keys:
          - embeddings: list[list[float]]  shape (n_samples, n_features)
          - labels: list[int]              shape (n_samples,)
          - group_labels: list[int]        shape (n_samples,)
          - description: str
    """
    from shortcut_detect import (
        generate_linear_shortcut,
        generate_no_shortcut,
        generate_nonlinear_shortcut,
    )

    rng = np.random.default_rng(seed)

    if shortcut_type == "linear":
        embeddings, labels = generate_linear_shortcut(
            n_samples=n_samples,
            embedding_dim=n_features,
            seed=seed,
        )
    elif shortcut_type == "nonlinear":
        embeddings, labels = generate_nonlinear_shortcut(
            n_samples=n_samples,
            embedding_dim=n_features,
            seed=seed,
        )
    elif shortcut_type == "none":
        embeddings, labels = generate_no_shortcut(
            n_samples=n_samples,
            embedding_dim=n_features,
            seed=seed,
        )
    else:
        return json.dumps(
            {
                "error": f"Unknown shortcut_type '{shortcut_type}'. Use 'linear', 'nonlinear', or 'none'."
            }
        )

    # Synthesize correlated group labels based on shortcut_strength
    flip_mask = rng.random(n_samples) > shortcut_strength
    group_labels = labels.copy()
    group_labels[flip_mask] = 1 - group_labels[flip_mask]

    return json.dumps(
        {
            "embeddings": embeddings.tolist(),
            "labels": labels.tolist(),
            "group_labels": group_labels.tolist(),
            "description": (
                f"{shortcut_type} shortcut dataset: {n_samples} samples, "
                f"{n_features} features, strength={shortcut_strength}, seed={seed}"
            ),
        }
    )


# ---------------------------------------------------------------------------
# Tool: run_detector
# ---------------------------------------------------------------------------


@mcp.tool()
def run_detector(
    embeddings: list[list[float]],
    labels: list[int],
    group_labels: list[int] | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
    session_id: str = "default",
) -> str:
    """Run shortcut detection on embedding data.

    Args:
        embeddings: 2-D array of embeddings, shape (n_samples, n_features).
                    Pass as a list of lists, e.g. [[0.1, 0.2], [0.3, 0.4]].
        labels: 1-D integer array of class labels, shape (n_samples,).
        group_labels: 1-D integer array of protected-group labels (optional).
                      Required for fairness methods (equalized_odds,
                      demographic_parity, geometric, statistical, hbac).
        methods: List of method names to run. Defaults to ["hbac", "probe", "statistical"].
                 Call list_methods() to see all available options.
        seed: Random seed for reproducibility (default 42).
        session_id: Identifier to cache results for later use with
                    get_summary() or compare_methods(). Default "default".

    Returns:
        JSON with keys:
          - session_id: str
          - methods_run: list[str]
          - shortcut_detected: bool  (overall verdict)
          - risk_level: str          ("low" | "moderate" | "high" | "unknown")
          - per_method: dict[str, {shortcut_detected, risk_level, notes}]
          - summary: str             (human-readable paragraph)
    """
    from shortcut_detect import ShortcutDetector

    emb = _parse_array(embeddings, "embeddings")
    lbl = _parse_array(labels, "labels").astype(int)
    grp = _parse_array(group_labels, "group_labels")
    if grp is not None:
        grp = grp.astype(int)

    if methods is None:
        methods = ["hbac", "probe", "statistical"]

    try:
        detector = ShortcutDetector(methods=methods, seed=seed)
        detector.fit(embeddings=emb, labels=lbl, group_labels=grp)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})

    raw = detector.get_results()
    _LAST_RESULTS[session_id] = {"detector": detector, "raw": raw}

    # Build per-method summary
    per_method: dict[str, Any] = {}
    any_detected = False
    worst_risk = "low"
    _risk_order = {"low": 0, "moderate": 1, "high": 2, "unknown": -1}

    for method, result in raw.items():
        if not isinstance(result, dict):
            continue
        detected = result.get("shortcut_detected")
        risk = result.get("risk_level", "unknown")
        notes = result.get("notes", "")

        # Handle multi-attribute results
        if "by_attribute" in result:
            sub: dict[str, Any] = {}
            for attr, attr_result in result["by_attribute"].items():
                sub[attr] = {
                    "shortcut_detected": attr_result.get("shortcut_detected"),
                    "risk_level": attr_result.get("risk_level", "unknown"),
                    "notes": attr_result.get("notes", ""),
                }
                if attr_result.get("shortcut_detected"):
                    any_detected = True
                ar = attr_result.get("risk_level", "low")
                if _risk_order.get(ar, -1) > _risk_order.get(worst_risk, 0):
                    worst_risk = ar
            per_method[method] = {"by_attribute": sub}
        else:
            per_method[method] = {
                "shortcut_detected": detected,
                "risk_level": risk,
                "notes": notes,
            }
            if detected:
                any_detected = True
            if _risk_order.get(risk, -1) > _risk_order.get(worst_risk, 0):
                worst_risk = risk

    summary = detector.summary()

    return json.dumps(
        _safe_serialize(
            {
                "session_id": session_id,
                "methods_run": list(raw.keys()),
                "shortcut_detected": any_detected,
                "risk_level": worst_risk,
                "per_method": per_method,
                "summary": summary,
            }
        ),
        indent=2,
    )


# ---------------------------------------------------------------------------
# Tool: get_summary
# ---------------------------------------------------------------------------


@mcp.tool()
def get_summary(session_id: str = "default") -> str:
    """Get a detailed human-readable summary for a previous run_detector call.

    Args:
        session_id: The session_id used in run_detector (default "default").

    Returns:
        Plain-text summary generated by ShortcutDetector.summary().
    """
    if session_id not in _LAST_RESULTS:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    return _LAST_RESULTS[session_id]["detector"].summary()


# ---------------------------------------------------------------------------
# Tool: get_method_detail
# ---------------------------------------------------------------------------


@mcp.tool()
def get_method_detail(method: str, session_id: str = "default") -> str:
    """Get full raw results for a single method from a previous run_detector call.

    Args:
        method: Method name (e.g. "hbac", "probe").
        session_id: The session_id used in run_detector (default "default").

    Returns:
        JSON with the complete result dict for that method.
    """
    if session_id not in _LAST_RESULTS:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    raw = _LAST_RESULTS[session_id]["raw"]
    if method not in raw:
        available = list(raw.keys())
        return json.dumps({"error": f"Method '{method}' not found. Available: {available}"})
    return json.dumps(_safe_serialize(raw[method]), indent=2)


# ---------------------------------------------------------------------------
# Tool: compare_methods
# ---------------------------------------------------------------------------


@mcp.tool()
def compare_methods(
    embeddings: list[list[float]],
    labels: list[int],
    group_labels: list[int] | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
) -> str:
    """Run multiple detectors and return a side-by-side comparison table.

    Args:
        embeddings: 2-D array of embeddings as list of lists.
        labels: Class labels as list of ints.
        group_labels: Protected-group labels as list of ints (optional).
        methods: Methods to compare. Defaults to
                 ["hbac", "probe", "statistical", "geometric", "demographic_parity"].
        seed: Random seed (default 42).

    Returns:
        JSON with keys:
          - table: list of {method, shortcut_detected, risk_level, notes}
          - consensus: "shortcut" | "no_shortcut" | "mixed"
          - high_risk_methods: list[str]
    """
    from shortcut_detect import ShortcutDetector

    if methods is None:
        methods = ["hbac", "probe", "statistical", "geometric", "demographic_parity"]

    emb = _parse_array(embeddings, "embeddings")
    lbl = _parse_array(labels, "labels").astype(int)
    grp = _parse_array(group_labels, "group_labels")
    if grp is not None:
        grp = grp.astype(int)

    try:
        detector = ShortcutDetector(methods=methods, seed=seed)
        detector.fit(embeddings=emb, labels=lbl, group_labels=grp)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})

    raw = detector.get_results()
    table = []
    high_risk = []
    detected_count = 0
    not_detected_count = 0

    for method, result in raw.items():
        if not isinstance(result, dict):
            continue
        if "by_attribute" in result:
            for attr, attr_result in result["by_attribute"].items():
                d = attr_result.get("shortcut_detected")
                r = attr_result.get("risk_level", "unknown")
                row = {
                    "method": f"{method}[{attr}]",
                    "shortcut_detected": d,
                    "risk_level": r,
                    "notes": attr_result.get("notes", ""),
                }
                table.append(row)
                if d:
                    detected_count += 1
                elif d is False:
                    not_detected_count += 1
                if r == "high":
                    high_risk.append(f"{method}[{attr}]")
        else:
            d = result.get("shortcut_detected")
            r = result.get("risk_level", "unknown")
            row = {
                "method": method,
                "shortcut_detected": d,
                "risk_level": r,
                "notes": result.get("notes", ""),
            }
            table.append(row)
            if d:
                detected_count += 1
            elif d is False:
                not_detected_count += 1
            if r == "high":
                high_risk.append(method)

    if detected_count > not_detected_count:
        consensus = "shortcut"
    elif not_detected_count > detected_count:
        consensus = "no_shortcut"
    else:
        consensus = "mixed"

    return json.dumps(
        _safe_serialize(
            {
                "table": table,
                "consensus": consensus,
                "high_risk_methods": high_risk,
                "votes": {"detected": detected_count, "not_detected": not_detected_count},
            }
        ),
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
