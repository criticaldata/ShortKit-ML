"""Tests for the ShortKit-ML MCP server tools."""

import asyncio
import json
from pathlib import Path

import numpy as np

import shortcut_detect.mcp_server as mcp_server
from shortcut_detect.mcp_server import (
    compare_methods,
    generate_synthetic_data,
    generate_report,
    get_method_detail,
    get_summary,
    list_methods,
    run_benchmark,
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


def test_generate_synthetic_data_strength_affects_none_embeddings():
    weak = json.loads(
        generate_synthetic_data(
            n_samples=40,
            n_features=6,
            shortcut_type="none",
            shortcut_strength=0.0,
            seed=7,
        )
    )
    strong = json.loads(
        generate_synthetic_data(
            n_samples=40,
            n_features=6,
            shortcut_type="none",
            shortcut_strength=1.0,
            seed=7,
        )
    )
    weak_arr = np.asarray(weak["embeddings"])
    strong_arr = np.asarray(strong["embeddings"])
    assert not np.allclose(weak_arr, strong_arr)


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
    assert result["shortcut_detected"] is True
    assert result["risk_level"] in {"moderate", "high"}


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


def test_run_detector_defaults_to_legacy_default_session():
    result = json.loads(
        run_detector(
            embeddings=_embeddings,
            labels=_labels,
            methods=["probe"],
        )
    )
    assert result["session_id"] == "default"


def test_run_detector_default_session_supports_follow_up_calls():
    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        group_labels=_groups,
        methods=["probe"],
    )
    summary = get_summary()
    detail = json.loads(get_method_detail("probe"))
    assert "Probe-based Detection" in summary
    assert isinstance(detail, dict)


def test_run_detector_accepts_file_inputs(tmp_path):
    embeddings_path = tmp_path / "embeddings.npy"
    labels_path = tmp_path / "labels.csv"
    groups_path = tmp_path / "groups.csv"

    np.save(embeddings_path, np.array(_embeddings))
    np.savetxt(labels_path, np.array(_labels, dtype=int), delimiter=",", fmt="%d", header="label", comments="")
    np.savetxt(groups_path, np.array(_groups, dtype=int), delimiter=",", fmt="%d", header="group", comments="")

    result = json.loads(
        run_detector(
            embeddings_path=str(embeddings_path),
            labels_path=str(labels_path),
            group_labels_path=str(groups_path),
            methods=["probe", "statistical"],
            session_id="file_input_session",
        )
    )

    assert "error" not in result
    assert result["methods_run"] == ["probe", "statistical"]


def test_run_detector_accepts_headerless_csv_inputs(tmp_path):
    embeddings_path = tmp_path / "embeddings.csv"
    labels_path = tmp_path / "labels.csv"
    groups_path = tmp_path / "groups.csv"

    np.savetxt(embeddings_path, np.array(_embeddings), delimiter=",")
    np.savetxt(labels_path, np.array(_labels, dtype=int), delimiter=",", fmt="%d")
    np.savetxt(groups_path, np.array(_groups, dtype=int), delimiter=",", fmt="%d")

    result = json.loads(
        run_detector(
            embeddings_path=str(embeddings_path),
            labels_path=str(labels_path),
            group_labels_path=str(groups_path),
            methods=["probe", "statistical"],
            session_id="headerless_file_input_session",
        )
    )

    assert "error" not in result
    assert result["methods_run"] == ["probe", "statistical"]


def test_run_detector_accepts_single_feature_embedding_file(tmp_path):
    embeddings_path = tmp_path / "embeddings.csv"
    labels_path = tmp_path / "labels.csv"

    single_feature_embeddings = np.array(_embeddings)[:, :1]
    np.savetxt(embeddings_path, single_feature_embeddings, delimiter=",")
    np.savetxt(labels_path, np.array(_labels, dtype=int), delimiter=",", fmt="%d")

    result = json.loads(
        run_detector(
            embeddings_path=str(embeddings_path),
            labels_path=str(labels_path),
            methods=["probe"],
            session_id="single_feature_file_input_session",
        )
    )

    assert "error" not in result
    assert result["methods_run"] == ["probe"]


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


def test_get_summary_uses_persistent_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "_CACHE_DIR", tmp_path)
    mcp_server._LAST_RESULTS.clear()

    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        group_labels=_groups,
        methods=["probe"],
        session_id="persistent_summary",
    )
    mcp_server._LAST_RESULTS.clear()

    summary = get_summary("persistent_summary")
    assert "Probe-based Detection" in summary


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
    assert result["votes"]["detected"] >= 1


def test_generate_report_creates_html(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "_CACHE_DIR", tmp_path / "cache")
    mcp_server._LAST_RESULTS.clear()

    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        group_labels=_groups,
        methods=["probe"],
        session_id="report_session",
    )
    output_path = tmp_path / "report.html"
    result = json.loads(
        generate_report(
            session_id="report_session",
            output_path=str(output_path),
            format="html",
            include_visualizations=False,
        )
    )

    assert result["output_path"] == str(output_path)
    assert output_path.exists()
    assert "html" in output_path.read_text(encoding="utf-8").lower()


def test_generate_report_with_csv_survives_persistent_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "_CACHE_DIR", tmp_path / "cache")
    mcp_server._LAST_RESULTS.clear()

    run_detector(
        embeddings=_embeddings,
        labels=_labels,
        group_labels=_groups,
        methods=["probe", "hbac"],
        session_id="persisted_report_session",
    )
    mcp_server._LAST_RESULTS.clear()

    output_path = tmp_path / "persisted_report.html"
    csv_dir = tmp_path / "persisted_csv"
    result = json.loads(
        generate_report(
            session_id="persisted_report_session",
            output_path=str(output_path),
            format="html",
            include_visualizations=False,
            export_csv=True,
            csv_dir=str(csv_dir),
        )
    )

    assert result["output_path"] == str(output_path)
    assert result["csv_dir"] == str(csv_dir)
    assert output_path.exists()
    assert csv_dir.exists()
    assert any(csv_dir.iterdir())


def test_generate_report_pdf_fallback_returns_actual_html_path(tmp_path, monkeypatch):
    class FakeDetector:
        def generate_report(self, output_path, format, include_visualizations, export_csv, csv_dir):
            fallback_path = Path(output_path).with_suffix(".html")
            fallback_path.write_text("<html>fallback</html>", encoding="utf-8")

    monkeypatch.setattr(mcp_server, "_build_detector_from_session", lambda session_id, owner_key=None: FakeDetector())

    output_path = tmp_path / "report.pdf"
    result = json.loads(
        generate_report(
            session_id="fallback_report_session",
            output_path=str(output_path),
            format="pdf",
            include_visualizations=False,
            return_base64=True,
        )
    )

    assert result["format"] == "html"
    assert result["output_path"] == str(output_path.with_suffix(".html"))
    assert result["content_base64"]


def test_run_benchmark_returns_artifacts(tmp_path):
    result = json.loads(
        run_benchmark(
            config={
                "benchmark_name": "test_benchmark",
                "methods": ["probe", "statistical"],
                "datasets": {
                    "synthetic": {
                        "enabled": True,
                        "n_seeds": 1,
                        "n_samples": 80,
                        "embedding_dim": 8,
                        "shortcut_dims": 2,
                        "effect_size": 0.8,
                    },
                    "chest_xray": {"enabled": False, "n_seeds": 0},
                },
                "output_dir": str(tmp_path / "benchmark"),
            }
        )
    )

    assert "error" not in result
    assert result["n_runs"] == 2
    assert Path(result["aggregate_path"]).exists()
    assert Path(result["manifest_path"]).exists()


def test_compare_methods_async_does_not_rerun_methods(monkeypatch):
    calls: list[str] = []

    def fake_run_single_method(method, **kwargs):
        calls.append(method)
        return {
            "shortcut_detected": method == "probe",
            "risk_level": "high" if method == "probe" else "low",
            "notes": f"called {method}",
        }

    monkeypatch.setattr(mcp_server, "_run_single_method", fake_run_single_method)

    result = json.loads(
        asyncio.run(
            mcp_server._compare_methods_async(
                embeddings=_embeddings,
                labels=_labels,
                group_labels=_groups,
                embeddings_path=None,
                labels_path=None,
                group_labels_path=None,
                methods=["probe", "statistical", "geometric"],
                seed=42,
                ctx=None,
            )
        )
    )

    assert calls == ["probe", "statistical", "geometric"]
    assert result["votes"] == {"detected": 1, "not_detected": 2}
