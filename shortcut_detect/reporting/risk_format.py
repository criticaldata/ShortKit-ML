"""Risk formatting helpers for consistent detector output presentation."""

from __future__ import annotations

from typing import Any

RISK_LEVELS = {"low", "moderate", "high", "unknown"}
SCOPED_METHODS = {
    "hbac",
    "probe",
    "frequency",
    "statistical",
    "geometric",
    "equalized_odds",
    "demographic_parity",
    "intersectional",
    "early_epoch_clustering",
    "bias_direction_pca",
    "gce",
    "cav",
    "sis",
    "causal_effect",
    "vae",
}


def normalize_risk_level(raw: str | None) -> str:
    value = str(raw or "unknown").strip().lower()
    if value == "medium":
        value = "moderate"
    if value in RISK_LEVELS:
        return value
    return "unknown"


def display_risk(level: str) -> str:
    value = normalize_risk_level(level)
    return "Unknown" if value == "unknown" else value.title()


def risk_css_class(level: str) -> str:
    return f"risk-{normalize_risk_level(level)}"


_RISK_ORDER = {"low": 0, "moderate": 1, "high": 2, "unknown": -1}


def apply_standardized_risk(method: str, result: dict[str, Any]) -> dict[str, Any]:
    """Attach standardized risk fields and summary header lines in-place."""
    if method not in SCOPED_METHODS:
        return result
    if not isinstance(result, dict):
        return result

    if "by_attribute" in result:
        by_attr = result["by_attribute"]
        worst_level = "low"
        worst_reason = ""
        for _attr_name, sub in by_attr.items():
            apply_standardized_risk(method, sub)
            rl = sub.get("risk_value") or sub.get("risk_level") or "unknown"
            rl = normalize_risk_level(rl)
            if _RISK_ORDER.get(rl, -1) > _RISK_ORDER.get(worst_level, -1):
                worst_level = rl
                worst_reason = sub.get("risk_reason", "")
        result["risk_value"] = worst_level
        result["risk_label"] = display_risk(worst_level)
        result["risk_reason"] = (
            f"Worst across attributes: {worst_reason}"
            if worst_reason
            else f"Multi-attribute analysis across {len(by_attr)} attributes."
        )
        return result

    if not result.get("success"):
        return result

    payload = build_method_risk(method, result)
    result.update(payload)

    summary_lines = result.get("summary_lines") or []
    if summary_lines:
        result["summary_lines"] = _standardize_summary_lines(summary_lines, payload)
    return result


def build_method_risk(method: str, result: dict[str, Any]) -> dict[str, str]:
    level = "unknown"
    reason = _fallback_reason(result)

    if method == "hbac":
        report = result.get("report") or {}
        shortcut_info = report.get("has_shortcut", {})
        confidence = normalize_risk_level(shortcut_info.get("confidence"))
        exists = shortcut_info.get("exists")
        evidence = shortcut_info.get("evidence", {})

        level = "low" if exists is False else confidence
        clusters = evidence.get("high_purity_clusters")
        linear_acc = evidence.get("linear_test_accuracy")
        reason = (
            f"Shortcut confidence '{display_risk(confidence)}'"
            f" with {clusters if clusters is not None else 'unknown'} high-purity clusters"
            f" and linear accuracy {_fmt_num(linear_acc)}."
        )

    elif method == "probe":
        probe_results = result.get("results") or {}
        metrics = probe_results.get("metrics", {})
        metric_name = metrics.get("metric", "metric")
        metric_value = metrics.get("metric_value")
        threshold = metrics.get("threshold")
        level = normalize_risk_level(probe_results.get("risk_level"))
        reason = (
            f"{metric_name} ({_fmt_num(metric_value)}) compared against threshold "
            f"({_fmt_num(threshold)})."
        )

    elif method == "frequency":
        report = result.get("report") or {}
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
        level = normalize_risk_level(report.get("risk_level", result.get("risk_level")))
        n_shortcut = metrics.get("n_shortcut_classes")
        n_classes = metrics.get("n_classes")
        tpr = metrics.get("tpr_threshold")
        fpr = metrics.get("fpr_threshold")
        reason = (
            f"{_fmt_num(n_shortcut)}/{_fmt_num(n_classes)} classes exceeded "
            f"TPR/FPR thresholds ({_fmt_num(tpr)}/{_fmt_num(fpr)})."
        )

    elif method == "statistical":
        alpha = result.get("alpha")
        correction = str(result.get("correction_method", "unknown")).upper()
        significant = result.get("significant_features") or {}
        n_sig = sum(1 for v in significant.values() if v is not None and len(v) > 0)
        n_total = len(significant)
        level = "moderate" if n_sig > 0 else "low"
        reason = (
            f"{n_sig}/{n_total} comparisons have significant features after "
            f"{correction} (alpha={_fmt_num(alpha)})."
        )

    elif method == "geometric":
        summary = result.get("summary") or {}
        level = normalize_risk_level(summary.get("risk_level"))
        reason = str(summary.get("message") or _fallback_reason(result))

    elif method == "equalized_odds":
        report = result.get("report")
        level = normalize_risk_level(_get(report, "risk_level"))
        tpr_gap = _get(report, "tpr_gap")
        fpr_gap = _get(report, "fpr_gap")
        detector = result.get("detector")
        tpr_thr = _get(detector, "tpr_gap_threshold")
        fpr_thr = _get(detector, "fpr_gap_threshold")
        threshold = (
            max(v for v in [tpr_thr, fpr_thr] if isinstance(v, int | float))
            if isinstance(tpr_thr, int | float) or isinstance(fpr_thr, int | float)
            else None
        )
        max_gap = (
            max(v for v in [tpr_gap, fpr_gap] if isinstance(v, int | float))
            if isinstance(tpr_gap, int | float) or isinstance(fpr_gap, int | float)
            else None
        )
        reason = (
            f"Max equalized-odds gap ({_fmt_num(max_gap)}) compared against threshold "
            f"({_fmt_num(threshold)})."
        )

    elif method == "demographic_parity":
        report = result.get("report")
        level = normalize_risk_level(_get(report, "risk_level"))
        dp_gap = _get(report, "dp_gap")
        detector = result.get("detector")
        threshold = _get(detector, "dp_gap_threshold")
        reason = f"DP gap ({_fmt_num(dp_gap)}) compared against threshold ({_fmt_num(threshold)})."

    elif method == "intersectional":
        report = result.get("report")
        level = normalize_risk_level(_get(report, "risk_level"))
        tpr_gap = _get(report, "tpr_gap")
        fpr_gap = _get(report, "fpr_gap")
        dp_gap = _get(report, "dp_gap")
        attrs = _get(report, "attribute_names") or []
        valid_gaps = [
            v for v in [tpr_gap, fpr_gap, dp_gap] if isinstance(v, int | float) and v == v
        ]
        max_gap = max(valid_gaps) if valid_gaps else None
        reason = (
            f"Max intersectional gap ({_fmt_num(max_gap)}) across attributes "
            f"({', '.join(attrs) if attrs else 'N/A'})."
        )

    elif method == "early_epoch_clustering":
        report = result.get("report")
        level = normalize_risk_level(_get(report, "risk_level"))
        gap = _get(report, "largest_gap")
        minority_ratio = _get(report, "minority_ratio")
        entropy = _get(report, "size_entropy")
        reason = (
            f"Largest gap={_fmt_num(gap)}, minority ratio={_fmt_num(minority_ratio)}, "
            f"entropy={_fmt_num(entropy)}."
        )

    elif method == "bias_direction_pca":
        report = result.get("report")
        level = normalize_risk_level(_get(report, "risk_level"))
        gap = _get(report, "projection_gap")
        threshold = _get(result.get("detector"), "gap_threshold")
        reason = (
            f"Projection gap ({_fmt_num(gap)}) compared against threshold "
            f"({_fmt_num(threshold)})."
        )

    elif method == "gce":
        report = result.get("report")
        level = normalize_risk_level(_get(report, "risk_level"))
        n_minority = _get(report, "n_minority")
        ratio = _get(report, "minority_ratio")
        threshold = _get(report, "threshold")
        reason = (
            f"Flagged {_fmt_num(n_minority)} high-loss samples "
            f"({_fmt_num(ratio)}) at loss percentile threshold {_fmt_num(threshold)}."
        )
    elif method == "cav":
        metrics = result.get("metrics") or {}
        report = result.get("report") or {}
        per_concept = report.get("per_concept", [])
        n_flagged = sum(1 for row in per_concept if isinstance(row, dict) and row.get("flagged"))
        n_tested = metrics.get("n_tested")
        max_tcav = metrics.get("max_tcav_score")
        max_quality = metrics.get("max_concept_quality")

        shortcut = result.get("shortcut_detected")
        if shortcut is None:
            level = "unknown"
        elif shortcut:
            level = "high" if isinstance(max_tcav, int | float) and max_tcav >= 0.75 else "moderate"
        else:
            level = "low"

        reason = (
            f"{_fmt_num(n_flagged)} flagged concepts out of {_fmt_num(n_tested)} tested, "
            f"max TCAV {_fmt_num(max_tcav)}, max concept quality {_fmt_num(max_quality)}."
        )

    elif method == "sis":
        metrics = result.get("metrics") or {}
        report = result.get("report") or {}
        mean_sis = metrics.get("mean_sis_size")
        frac_dim = metrics.get("frac_dimensions")
        n_computed = metrics.get("n_computed")
        threshold = result.get("detector")
        threshold = getattr(threshold, "shortcut_threshold", 0.15) if threshold else 0.15

        shortcut = result.get("shortcut_detected")
        if shortcut is None:
            level = "unknown"
        elif shortcut:
            level = "high" if isinstance(frac_dim, int | float) and frac_dim <= 0.1 else "moderate"
        else:
            level = "low"

        reason = (
            f"Mean SIS size {_fmt_num(mean_sis)} ({_fmt_num(frac_dim)} of dims) "
            f"from {_fmt_num(n_computed)} samples (threshold {_fmt_num(threshold)})."
        )

    elif method == "causal_effect":
        metrics = result.get("metrics") or {}
        n_attributes = metrics.get("n_attributes")
        n_spurious = metrics.get("n_spurious")
        threshold = metrics.get("spurious_threshold")

        shortcut = result.get("shortcut_detected")
        if shortcut is None:
            level = "unknown"
        elif shortcut:
            level = "high" if n_spurious and n_spurious > 1 else "moderate"
        else:
            level = "low"

        reason = (
            f"{_fmt_num(n_spurious)} spurious attribute(s) out of {_fmt_num(n_attributes)} "
            f"(threshold {_fmt_num(threshold)})."
        )

    elif method == "vae":
        metrics = result.get("metrics") or {}
        n_flagged = metrics.get("n_flagged", 0)
        max_pred = metrics.get("max_predictiveness")
        latent_dim = metrics.get("latent_dim")

        shortcut = result.get("shortcut_detected")
        if shortcut is None:
            level = "unknown"
        elif shortcut:
            level = "high" if n_flagged and n_flagged >= (latent_dim or 0) // 2 else "moderate"
        else:
            level = "low"

        reason = (
            f"{_fmt_num(n_flagged)} latent dimension(s) flagged as shortcut candidates "
            f"out of {_fmt_num(latent_dim)}, max predictiveness {_fmt_num(max_pred)}."
        )

    level = normalize_risk_level(level)
    return {
        "risk_value": level,
        "risk_label": display_risk(level),
        "risk_reason": reason,
    }


def _standardize_summary_lines(summary_lines: list[str], payload: dict[str, str]) -> list[str]:
    filtered: list[str] = []
    for line in summary_lines:
        if not isinstance(line, str):
            continue
        stripped = line.strip()
        lowered = stripped.lower()
        if (
            lowered.startswith("risk level:")
            or lowered.startswith("confidence:")
            or lowered.startswith("assessment:")
        ):
            continue

        if "| risk:" in lowered:
            idx = lowered.index("| risk:")
            stripped = stripped[:idx].rstrip()
        elif " (risk:" in lowered:
            idx = lowered.index(" (risk:")
            stripped = stripped[:idx].rstrip()

        if stripped:
            filtered.append(stripped)

    return [f"Risk: {payload['risk_label']}", f"Reason: {payload['risk_reason']}", *filtered]


def _fallback_reason(result: dict[str, Any]) -> str:
    for key in ("risk_reason", "notes"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    report = result.get("report")
    notes = _get(report, "notes")
    if isinstance(notes, str) and notes.strip():
        return notes.strip()
    return "Insufficient information to determine a specific risk trigger."


def _fmt_num(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "unknown"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
