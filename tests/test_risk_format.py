"""Tests for standardized risk formatting helpers."""

from shortcut_detect.reporting.risk_format import (
    apply_standardized_risk,
    build_method_risk,
    normalize_risk_level,
)


def test_normalize_risk_level_maps_medium_to_moderate():
    assert normalize_risk_level("medium") == "moderate"
    assert normalize_risk_level("Moderate") == "moderate"


def test_build_method_risk_probe_uses_metric_and_threshold():
    result = {
        "success": True,
        "results": {
            "risk_level": "high",
            "metrics": {
                "metric": "f1",
                "metric_value": 0.82,
                "threshold": 0.70,
            },
        },
    }
    payload = build_method_risk("probe", result)
    assert payload["risk_label"] == "High"
    assert "f1" in payload["risk_reason"]
    assert "0.700" in payload["risk_reason"]


def test_apply_standardized_risk_prepends_risk_lines_and_removes_legacy_tokens():
    result = {
        "success": True,
        "report": {
            "has_shortcut": {
                "exists": True,
                "confidence": "medium",
                "evidence": {
                    "high_purity_clusters": 2,
                    "linear_test_accuracy": 0.88,
                },
            }
        },
        "summary_lines": ["Confidence: medium", "Clusters found: 4"],
    }

    apply_standardized_risk("hbac", result)

    assert result["summary_lines"][0].startswith("Risk: ")
    assert result["summary_lines"][1].startswith("Reason: ")
    assert all(not line.startswith("Confidence:") for line in result["summary_lines"])


def test_build_method_risk_cav_uses_tcav_and_quality():
    result = {
        "success": True,
        "shortcut_detected": True,
        "metrics": {
            "n_tested": 2,
            "max_tcav_score": 0.81,
            "max_concept_quality": 0.9,
        },
        "report": {
            "per_concept": [
                {"concept_name": "a", "flagged": True},
                {"concept_name": "b", "flagged": False},
            ]
        },
    }
    payload = build_method_risk("cav", result)
    assert payload["risk_label"] == "High"
    assert "max TCAV" in payload["risk_reason"]
