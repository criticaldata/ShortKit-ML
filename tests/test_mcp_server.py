"""Tests for the ShortKit-ML MCP server tools."""

import json

import numpy as np

from shortcut_detect.mcp_server import (
    compare_methods,
    generate_synthetic_data,
    get_method_detail,
    get_summary,
    list_methods,
    run_detector,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 100
N_FEATURES = 16
RNG = np.random.default_rng(42)

_embeddings = RNG.standard_normal((N_SAMPLES, N_FEATURES)).tolist()
_labels = (RNG.integers(0, 2, N_SAMPLES)).tolist()
_groups = (RNG.integers(0, 2, N_SAMPLES)).tolist()


# ---------------------------------------------------------------------------
# list_methods
# ---------------------------------------------------------------------------


def test_list_methods_returns_valid_json():
    result = list_methods()
    data = json.loads(result)
    assert isinstance(data, dict)
    assert len(data) > 0


def test_list_methods_contains_core_methods():
    data = json.loads(list_methods())
    for method in ["hbac", "probe", "statistical", "geometric"]:
        assert method in data, f"Expected '{method}' in list_methods output"


def test_list_methods_values_are_strings():
    data = json.loads(list_methods())
    for k, v in data.items():
        assert isinstance(v, str), f"Description for '{k}' should be a string"


# ---------------------------------------------------------------------------
# generate_synthetic_data
# ---------------------------------------------------------------------------


def test_generate_synthetic_data_linear():
    result = json.loads(generate_synthetic_data(n_samples=80, n_features=8, shortcut_type="linear"))
    assert "embeddings" in result
    assert "labels" in result
    assert "group_labels" in result
    assert len(result["embeddings"]) == 80
    assert len(result["embeddings"][0]) == 8
    assert len(result["labels"]) == 80
    assert len(result["group_labels"]) == 80


def test_generate_synthetic_data_nonlinear():
    result = json.loads(
        generate_synthetic_data(n_samples=60, n_features=8, shortcut_type="nonlinear")
    )
    assert "embeddings" in result
    assert len(result["embeddings"]) == 60


def test_generate_synthetic_data_none():
    result = json.loads(generate_synthetic_data(n_samples=60, n_features=8, shortcut_type="none"))
    assert "embeddings" in result
    assert "error" not in result


def test_generate_synthetic_data_invalid_type():
    result = json.loads(generate_synthetic_data(shortcut_type="unknown_type"))
    assert "error" in result


def test_generate_synthetic_data_description():
    result = json.loads(generate_synthetic_data(shortcut_type="linear"))
    assert "description" in result
    assert "linear" in result["description"]


# ---------------------------------------------------------------------------
# run_detector
# ---------------------------------------------------------------------------


def test_run_detector_default_methods():
    result = json.loads(
        run_detector(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            session_id="test_default",
        )
    )
    assert "methods_run" in result
    assert "shortcut_detected" in result
    assert "risk_level" in result
    assert "per_method" in result
    assert "summary" in result


def test_run_detector_returns_valid_risk_level():
    result = json.loads(
        run_detector(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            session_id="test_risk",
        )
    )
    assert result["risk_level"] in {"low", "moderate", "high", "unknown"}


def test_run_detector_single_method_probe():
    result = json.loads(
        run_detector(
            embeddings=_embeddings,
            labels=_labels,
            methods=["probe"],
            session_id="test_probe",
        )
    )
    assert "probe" in result["methods_run"]
    assert "probe" in result["per_method"]


def test_run_detector_single_method_hbac():
    result = json.loads(
        run_detector(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            methods=["hbac"],
            session_id="test_hbac",
        )
    )
    assert "hbac" in result["methods_run"]


def test_run_detector_statistical_with_groups():
    result = json.loads(
        run_detector(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            methods=["statistical"],
            session_id="test_stat",
        )
    )
    assert "error" not in result
    assert "statistical" in result["methods_run"]


def test_run_detector_shortcut_detected_is_bool():
    result = json.loads(
        run_detector(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            session_id="test_bool",
        )
    )
    assert isinstance(result["shortcut_detected"], bool)


def test_run_detector_with_synthetic_shortcut_data():
    """MCP server correctly passes synthetic data through to detectors."""
    synth = json.loads(
        generate_synthetic_data(
            n_samples=200, n_features=16, shortcut_strength=0.95, shortcut_type="linear", seed=0
        )
    )
    result = json.loads(
        run_detector(
            embeddings=synth["embeddings"],
            labels=synth["labels"],
            group_labels=synth["group_labels"],
            methods=["probe", "statistical"],
            session_id="test_synth_shortcut",
        )
    )
    assert "error" not in result
    assert isinstance(result["shortcut_detected"], bool)
    assert result["risk_level"] in {"low", "moderate", "high", "unknown"}


def test_run_detector_session_id_isolation():
    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        methods=["probe"],
        session_id="session_a",
    )
    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        methods=["hbac"],
        group_labels=_groups,
        session_id="session_b",
    )
    summary_a = get_summary("session_a")
    summary_b = get_summary("session_b")
    assert isinstance(summary_a, str)
    assert isinstance(summary_b, str)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


def test_get_summary_returns_string():
    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        group_labels=_groups,
        session_id="test_summary",
    )
    summary = get_summary("test_summary")
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_get_summary_missing_session():
    result = get_summary("nonexistent_session_xyz")
    assert "No results found" in result


# ---------------------------------------------------------------------------
# get_method_detail
# ---------------------------------------------------------------------------


def test_get_method_detail_probe():
    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        methods=["probe"],
        session_id="test_detail",
    )
    detail = json.loads(get_method_detail("probe", session_id="test_detail"))
    assert isinstance(detail, dict)


def test_get_method_detail_missing_method():
    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        methods=["probe"],
        session_id="test_detail_missing",
    )
    detail = json.loads(get_method_detail("hbac", session_id="test_detail_missing"))
    assert "error" in detail


def test_get_method_detail_missing_session():
    result = get_method_detail("probe", session_id="no_such_session")
    assert "No results found" in result


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------


def test_compare_methods_returns_table():
    result = json.loads(
        compare_methods(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            methods=["probe", "statistical"],
        )
    )
    assert "table" in result
    assert "consensus" in result
    assert "high_risk_methods" in result
    assert "votes" in result


def test_compare_methods_consensus_is_valid():
    result = json.loads(
        compare_methods(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            methods=["probe", "statistical"],
        )
    )
    assert result["consensus"] in {"shortcut", "no_shortcut", "mixed"}


def test_compare_methods_table_structure():
    result = json.loads(
        compare_methods(
            embeddings=_embeddings,
            labels=_labels,
            group_labels=_groups,
            methods=["probe", "hbac"],
        )
    )
    for row in result["table"]:
        assert "method" in row
        assert "shortcut_detected" in row
        assert "risk_level" in row


def test_compare_methods_strong_shortcut_consensus():
    """With a very strong synthetic shortcut, consensus should be 'shortcut'."""
    synth = json.loads(
        generate_synthetic_data(
            n_samples=300, n_features=16, shortcut_strength=0.99, shortcut_type="linear", seed=1
        )
    )
    result = json.loads(
        compare_methods(
            embeddings=synth["embeddings"],
            labels=synth["labels"],
            group_labels=synth["group_labels"],
            methods=["probe", "statistical"],
        )
    )
    assert result["consensus"] in {"shortcut", "mixed"}
