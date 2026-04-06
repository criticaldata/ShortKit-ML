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

import asyncio
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import traceback
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("ShortKit-ML")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAST_RESULTS: dict[str, dict[str, Any]] = {}
_SESSION_LOCK = threading.RLock()
_CACHE_DIR = Path(
    os.environ.get(
        "SHORTKIT_MCP_CACHE_DIR",
        Path.home() / ".cache" / "shortkit-ml-mcp",
    )
)


def _cache_path_for_session(session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{digest}.joblib"


def _owner_key_from_ctx(ctx: Context | None) -> str | None:
    if ctx is None:
        return None
    client_id = ctx.client_id
    return str(client_id) if client_id else None


def _normalize_session_id(session_id: str | None, owner_key: str | None = None) -> str:
    if session_id:
        return session_id
    if owner_key is None:
        return "default"
    prefix = owner_key
    return f"{prefix}-{uuid.uuid4().hex}"


def _assert_session_access(
    session_id: str,
    *,
    owner_key: str | None,
    write: bool,
) -> None:
    session = _LAST_RESULTS.get(session_id)
    if session is None:
        path = _cache_path_for_session(session_id)
        if not path.exists():
            return
        session = joblib.load(path)
        _LAST_RESULTS[session_id] = session

    stored_owner = session.get("owner_key")
    if owner_key and stored_owner and stored_owner != owner_key:
        action = "write to" if write else "read"
        raise PermissionError(
            f"Session '{session_id}' belongs to a different MCP client; cannot {action} it."
        )


def _save_session(session_id: str, payload: dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final_path = _cache_path_for_session(session_id)
    fd, tmp_name = tempfile.mkstemp(prefix="shortkit-session-", suffix=".joblib", dir=_CACHE_DIR)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _load_session(session_id: str, owner_key: str | None = None) -> dict[str, Any] | None:
    with _SESSION_LOCK:
        _assert_session_access(session_id, owner_key=owner_key, write=False)
        cached = _LAST_RESULTS.get(session_id)
        if cached is not None:
            return cached

        path = _cache_path_for_session(session_id)
        if not path.exists():
            return None

        payload = joblib.load(path)
        _LAST_RESULTS[session_id] = payload
        return payload


def _safe_serialize(obj: Any) -> Any:
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
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _parse_array(data: list | np.ndarray | None, name: str) -> np.ndarray | None:
    if data is None:
        return None
    try:
        return np.array(data)
    except Exception as exc:
        raise ValueError(f"Could not parse '{name}' as numeric array: {exc}") from exc


def _load_array_from_path(path: str, name: str) -> np.ndarray:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"{name} file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return np.load(file_path)
    if suffix == ".npz":
        with np.load(file_path) as npz_file:
            keys = list(npz_file.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{name} .npz files must contain exactly one array, found keys={keys}"
                )
            return np.asarray(npz_file[keys[0]])
    if suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            raw = np.loadtxt(file_path, delimiter=delimiter, ndmin=2)
        except ValueError:
            frame = pd.read_csv(file_path, sep=delimiter)
            if frame.shape[1] == 1 and name != "embeddings":
                return frame.iloc[:, 0].to_numpy()
            return frame.to_numpy()
        if raw.shape[1] == 1 and name != "embeddings":
            return raw[:, 0]
        return raw
    raise ValueError(
        f"Unsupported {name} file format '{suffix}'. Use .npy, .npz, .csv, .tsv, or .txt."
    )


def _resolve_array_input(
    *,
    data: list | np.ndarray | None,
    path: str | None,
    name: str,
) -> np.ndarray | None:
    if data is not None:
        return _parse_array(data, name)
    if path:
        return _load_array_from_path(path, name)
    return None


def _extract_method_outcome(result: dict[str, Any]) -> tuple[Any, str, str]:
    if "shortcut_detected" in result or "risk_level" in result:
        detected = result.get("shortcut_detected")
        risk = str(result.get("risk_level", "unknown")).lower()
        notes = str(result.get("notes") or result.get("risk_reason") or "")
        return detected, risk, notes

    nested = result.get("results")
    if isinstance(nested, dict):
        detected = nested.get("shortcut_detected")
        risk = str(nested.get("risk_level") or result.get("risk_value") or "unknown").lower()
        notes = str(nested.get("notes") or result.get("risk_reason") or "")
        return detected, risk, notes

    report = result.get("report")
    if isinstance(report, dict) and isinstance(report.get("has_shortcut"), dict):
        detected = report["has_shortcut"].get("exists")
        risk = str(
            report["has_shortcut"].get("confidence")
            or result.get("risk_value")
            or "unknown"
        ).lower()
        notes = str(result.get("risk_reason") or "")
        return detected, risk, notes

    detected = result.get("success")
    risk = str(result.get("risk_value") or "unknown").lower()
    notes = str(result.get("risk_reason") or "")
    return detected, risk, notes


def _prepare_inputs(
    *,
    embeddings: list[list[float]] | np.ndarray | None,
    labels: list[int] | np.ndarray | None,
    group_labels: list[int] | np.ndarray | None,
    embeddings_path: str | None,
    labels_path: str | None,
    group_labels_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    emb = _resolve_array_input(data=embeddings, path=embeddings_path, name="embeddings")
    lbl = _resolve_array_input(data=labels, path=labels_path, name="labels")
    grp = _resolve_array_input(data=group_labels, path=group_labels_path, name="group_labels")
    if emb is None:
        raise ValueError("Provide embeddings inline or via embeddings_path.")
    if lbl is None:
        raise ValueError("Provide labels inline or via labels_path.")
    lbl = lbl.astype(int)
    if grp is not None:
        grp = grp.astype(int)
    return emb, lbl, grp


def _build_aggregate_detector(
    *,
    methods: list[str],
    seed: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray | None,
    raw_results: dict[str, Any],
):
    from shortcut_detect import ShortcutDetector

    detector = ShortcutDetector(methods=methods, seed=seed)
    detector.embeddings_ = embeddings
    detector.labels_ = labels
    detector.group_labels_ = group_labels
    detector.protected_labels_ = group_labels
    detector.results_ = raw_results
    detector.embedding_metadata_ = {"mode": "precomputed", "cached": False}
    detector.detectors_ = {
        method: result.get("detector")
        for method, result in raw_results.items()
        if isinstance(result, dict) and result.get("detector") is not None
    }
    return detector


def _run_single_method(
    method: str,
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray | None,
    seed: int,
) -> dict[str, Any]:
    from shortcut_detect import ShortcutDetector

    detector = ShortcutDetector(methods=[method], seed=seed)
    with contextlib.redirect_stdout(sys.stderr):
        detector.fit(embeddings=embeddings, labels=labels, group_labels=group_labels)
    return detector.results_.get(method, {"success": False, "error": "missing method result"})


def _summarize_per_method(raw: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    per_method: dict[str, Any] = {}
    any_detected = False
    worst_risk = "low"
    risk_order = {"low": 0, "moderate": 1, "high": 2, "unknown": -1}

    for method, result in raw.items():
        if not isinstance(result, dict):
            continue
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
                attr_risk = attr_result.get("risk_level", "low")
                if risk_order.get(attr_risk, -1) > risk_order.get(worst_risk, 0):
                    worst_risk = attr_risk
            per_method[method] = {"by_attribute": sub}
            continue

        detected, risk, notes = _extract_method_outcome(result)
        per_method[method] = {
            "shortcut_detected": detected,
            "risk_level": risk,
            "notes": notes,
        }
        if detected:
            any_detected = True
        if risk_order.get(risk, -1) > risk_order.get(worst_risk, 0):
            worst_risk = risk

    return per_method, any_detected, worst_risk


def _build_compare_methods_response(raw: dict[str, Any]) -> str:
    table = []
    high_risk = []
    detected_count = 0
    not_detected_count = 0

    for method, result in raw.items():
        if not isinstance(result, dict):
            continue
        if "by_attribute" in result:
            for attr, attr_result in result["by_attribute"].items():
                detected = attr_result.get("shortcut_detected")
                risk = attr_result.get("risk_level", "unknown")
                row = {
                    "method": f"{method}[{attr}]",
                    "shortcut_detected": detected,
                    "risk_level": risk,
                    "notes": attr_result.get("notes", ""),
                }
                table.append(row)
                if detected:
                    detected_count += 1
                elif detected is False:
                    not_detected_count += 1
                if risk == "high":
                    high_risk.append(f"{method}[{attr}]")
            continue

        detected, risk, notes = _extract_method_outcome(result)
        table.append(
            {
                "method": method,
                "shortcut_detected": detected,
                "risk_level": risk,
                "notes": notes,
            }
        )
        if detected:
            detected_count += 1
        elif detected is False:
            not_detected_count += 1
        if risk == "high":
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


def _store_session(
    *,
    session_id: str,
    owner_key: str | None,
    detector: Any,
    summary: str,
    raw: dict[str, Any],
    input_args: dict[str, Any],
) -> None:
    session_payload = {
        "owner_key": owner_key,
        "detector": detector,
        "summary": summary,
        "raw": raw,
        "input_args": input_args,
    }
    persisted = {
        "owner_key": owner_key,
        "summary": summary,
        "raw": raw,
        "input_args": input_args,
    }
    with _SESSION_LOCK:
        _assert_session_access(session_id, owner_key=owner_key, write=True)
        _LAST_RESULTS[session_id] = session_payload
        _save_session(session_id, persisted)


def _build_detector_from_session(session_id: str, owner_key: str | None = None):
    session = _load_session(session_id, owner_key=owner_key)
    if session is None:
        raise KeyError(session_id)

    detector = session.get("detector")
    if detector is not None:
        return detector

    input_args = session.get("input_args", {})
    detector = _build_aggregate_detector(
        methods=input_args.get("methods", []),
        seed=input_args.get("seed", 42),
        embeddings=input_args.get("embeddings"),
        labels=input_args.get("labels"),
        group_labels=input_args.get("group_labels"),
        raw_results=session["raw"],
    )
    session["detector"] = detector
    with _SESSION_LOCK:
        _LAST_RESULTS[session_id] = session
    return detector


def _run_detector_impl(
    *,
    embeddings: list[list[float]] | np.ndarray | None,
    labels: list[int] | np.ndarray | None,
    group_labels: list[int] | np.ndarray | None,
    embeddings_path: str | None,
    labels_path: str | None,
    group_labels_path: str | None,
    methods: list[str] | None,
    seed: int,
    session_id: str | None,
    owner_key: str | None,
) -> str:
    emb, lbl, grp = _prepare_inputs(
        embeddings=embeddings,
        labels=labels,
        group_labels=group_labels,
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        group_labels_path=group_labels_path,
    )
    run_methods = methods or ["hbac", "probe", "statistical"]
    final_session_id = _normalize_session_id(session_id, owner_key)

    raw: dict[str, Any] = {}
    for method in run_methods:
        raw[method] = _run_single_method(
            method,
            embeddings=emb,
            labels=lbl,
            group_labels=grp,
            seed=seed,
        )

    detector = _build_aggregate_detector(
        methods=run_methods,
        seed=seed,
        embeddings=emb,
        labels=lbl,
        group_labels=grp,
        raw_results=raw,
    )
    summary = detector.summary()
    per_method, any_detected, worst_risk = _summarize_per_method(raw)

    _store_session(
        session_id=final_session_id,
        owner_key=owner_key,
        detector=detector,
        summary=summary,
        raw=raw,
        input_args={
            "embeddings": emb,
            "labels": lbl,
            "group_labels": grp,
            "methods": run_methods,
            "seed": seed,
        },
    )

    return json.dumps(
        _safe_serialize(
            {
                "session_id": final_session_id,
                "methods_run": run_methods,
                "shortcut_detected": any_detected,
                "risk_level": worst_risk,
                "per_method": per_method,
                "summary": summary,
            }
        ),
        indent=2,
    )


async def _run_detector_async(
    *,
    embeddings: list[list[float]] | np.ndarray | None,
    labels: list[int] | np.ndarray | None,
    group_labels: list[int] | np.ndarray | None,
    embeddings_path: str | None,
    labels_path: str | None,
    group_labels_path: str | None,
    methods: list[str] | None,
    seed: int,
    session_id: str | None,
    owner_key: str | None,
    ctx: Context | None,
) -> str:
    emb, lbl, grp = _prepare_inputs(
        embeddings=embeddings,
        labels=labels,
        group_labels=group_labels,
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        group_labels_path=group_labels_path,
    )
    run_methods = methods or ["hbac", "probe", "statistical"]
    final_session_id = _normalize_session_id(session_id, owner_key)

    if ctx is not None:
        await ctx.info(f"Starting shortcut detection with {len(run_methods)} methods")
        await ctx.report_progress(0, 100, "Preparing detection run")

    raw: dict[str, Any] = {}
    total = max(len(run_methods), 1)
    for index, method in enumerate(run_methods, start=1):
        if ctx is not None:
            await ctx.report_progress(
                ((index - 1) / total) * 100,
                100,
                f"Running {method}",
            )
        raw[method] = await asyncio.to_thread(
            _run_single_method,
            method,
            embeddings=emb,
            labels=lbl,
            group_labels=grp,
            seed=seed,
        )
        if ctx is not None:
            await ctx.report_progress((index / total) * 100, 100, f"Finished {method}")

    detector = _build_aggregate_detector(
        methods=run_methods,
        seed=seed,
        embeddings=emb,
        labels=lbl,
        group_labels=grp,
        raw_results=raw,
    )
    summary = detector.summary()
    per_method, any_detected, worst_risk = _summarize_per_method(raw)

    await asyncio.to_thread(
        _store_session,
        session_id=final_session_id,
        owner_key=owner_key,
        detector=detector,
        summary=summary,
        raw=raw,
        input_args={
            "embeddings": emb,
            "labels": lbl,
            "group_labels": grp,
            "methods": run_methods,
            "seed": seed,
        },
    )

    if ctx is not None:
        await ctx.report_progress(100, 100, "Detection complete")

    return json.dumps(
        _safe_serialize(
            {
                "session_id": final_session_id,
                "methods_run": run_methods,
                "shortcut_detected": any_detected,
                "risk_level": worst_risk,
                "per_method": per_method,
                "summary": summary,
            }
        ),
        indent=2,
    )


def _compare_methods_impl(
    *,
    embeddings: list[list[float]] | np.ndarray | None,
    labels: list[int] | np.ndarray | None,
    group_labels: list[int] | np.ndarray | None,
    embeddings_path: str | None,
    labels_path: str | None,
    group_labels_path: str | None,
    methods: list[str] | None,
    seed: int,
) -> str:
    emb, lbl, grp = _prepare_inputs(
        embeddings=embeddings,
        labels=labels,
        group_labels=group_labels,
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        group_labels_path=group_labels_path,
    )
    run_methods = methods or ["hbac", "probe", "statistical", "geometric", "demographic_parity"]

    raw: dict[str, Any] = {}
    for method in run_methods:
        raw[method] = _run_single_method(
            method,
            embeddings=emb,
            labels=lbl,
            group_labels=grp,
            seed=seed,
        )

    return _build_compare_methods_response(raw)


async def _compare_methods_async(
    *,
    embeddings: list[list[float]] | np.ndarray | None,
    labels: list[int] | np.ndarray | None,
    group_labels: list[int] | np.ndarray | None,
    embeddings_path: str | None,
    labels_path: str | None,
    group_labels_path: str | None,
    methods: list[str] | None,
    seed: int,
    ctx: Context | None,
) -> str:
    emb, lbl, grp = _prepare_inputs(
        embeddings=embeddings,
        labels=labels,
        group_labels=group_labels,
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        group_labels_path=group_labels_path,
    )
    run_methods = methods or ["hbac", "probe", "statistical", "geometric", "demographic_parity"]

    if ctx is not None:
        await ctx.info(f"Comparing {len(run_methods)} methods")
        await ctx.report_progress(0, 100, "Preparing method comparison")

    raw: dict[str, Any] = {}
    total = max(len(run_methods), 1)
    for index, method in enumerate(run_methods, start=1):
        if ctx is not None:
            await ctx.report_progress(
                ((index - 1) / total) * 100,
                100,
                f"Running {method}",
            )
        raw[method] = await asyncio.to_thread(
            _run_single_method,
            method,
            embeddings=emb,
            labels=lbl,
            group_labels=grp,
            seed=seed,
        )
        if ctx is not None:
            await ctx.report_progress((index / total) * 100, 100, f"Finished {method}")

    if ctx is not None:
        await ctx.report_progress(100, 100, "Comparison complete")

    return _build_compare_methods_response(raw)


def _generate_report_impl(
    *,
    session_id: str,
    output_path: str | None,
    format: str,
    include_visualizations: bool,
    export_csv: bool,
    csv_dir: str | None,
    return_base64: bool,
    owner_key: str | None,
) -> str:
    detector = _build_detector_from_session(session_id, owner_key=owner_key)
    fmt = format.lower()

    if output_path is None:
        suffix = {"html": ".html", "pdf": ".pdf", "markdown": ".md"}.get(fmt, f".{fmt}")
        output_path = str(Path(tempfile.gettempdir()) / f"shortkit_report_{session_id}{suffix}")

    with contextlib.redirect_stdout(sys.stderr):
        detector.generate_report(
            output_path=output_path,
            format=fmt,
            include_visualizations=include_visualizations,
            export_csv=export_csv,
            csv_dir=csv_dir,
        )

    actual_output_path = Path(output_path)
    if fmt == "pdf" and not actual_output_path.exists():
        fallback_html_path = actual_output_path.with_suffix(".html")
        if fallback_html_path.exists():
            actual_output_path = fallback_html_path

    response: dict[str, Any] = {
        "session_id": session_id,
        "format": actual_output_path.suffix.lstrip(".") or fmt,
        "output_path": str(actual_output_path),
    }
    if export_csv:
        response["csv_dir"] = csv_dir or str(Path(output_path).parent / "csv_results")
    if return_base64:
        response["content_base64"] = base64.b64encode(actual_output_path.read_bytes()).decode(
            "ascii"
        )
    return json.dumps(response, indent=2)


def _run_benchmark_impl(
    *,
    config: dict[str, Any] | None,
    config_path: str | None,
) -> str:
    from shortcut_detect import BenchmarkConfig, BenchmarkRunner

    if config_path:
        benchmark_config = BenchmarkConfig.from_path(config_path)
    elif config is not None:
        benchmark_config = BenchmarkConfig.from_dict(config)
    else:
        benchmark_config = BenchmarkConfig.from_dict(
            {
                "benchmark_name": "mcp_smoke_benchmark",
                "methods": ["probe", "statistical", "hbac"],
                "primary_endpoint": "probe_metric_value",
                "random_seed": 42,
                "datasets": {
                    "synthetic": {
                        "enabled": True,
                        "n_seeds": 2,
                        "n_samples": 200,
                        "embedding_dim": 16,
                        "shortcut_dims": 5,
                        "effect_size": 1.0,
                    },
                    "chest_xray": {"enabled": False, "n_seeds": 0},
                },
                "output_dir": str(Path(tempfile.gettempdir()) / "shortkit_mcp_benchmark"),
            }
        )

    runner = BenchmarkRunner(benchmark_config)
    with contextlib.redirect_stdout(sys.stderr):
        result = runner.run()
    aggregate = result["aggregate"]
    preview = aggregate.head(10).replace({np.nan: None}).to_dict(orient="records")
    return json.dumps(
        {
            "output_dir": result["output_dir"],
            "runs_path": str(runner.runs_jsonl),
            "aggregate_path": str(runner.aggregate_csv),
            "primary_path": str(runner.primary_csv),
            "paired_tests_path": str(runner.paired_csv),
            "manifest_path": str(runner.manifest_json),
            "n_runs": int(result["runs"].shape[0]),
            "aggregate_preview": preview,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Public sync API (used in tests and direct Python examples)
# ---------------------------------------------------------------------------


def list_methods() -> str:
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


def generate_synthetic_data(
    n_samples: int = 200,
    n_features: int = 32,
    shortcut_strength: float = 0.8,
    shortcut_type: str = "linear",
    n_classes: int = 2,
    seed: int = 42,
) -> str:
    from shortcut_detect import (
        generate_linear_shortcut,
        generate_no_shortcut,
        generate_nonlinear_shortcut,
    )

    rng = np.random.default_rng(seed)
    shortcut_strength = float(np.clip(shortcut_strength, 0.0, 1.0))

    if shortcut_type == "linear":
        embeddings, labels = generate_linear_shortcut(
            n_samples=n_samples,
            embedding_dim=n_features,
            seed=seed,
        )
        noise = rng.standard_normal(size=embeddings.shape)
        embeddings = shortcut_strength * embeddings + (1.0 - shortcut_strength) * noise
        flip_mask = rng.random(n_samples) > shortcut_strength
        group_labels = labels.copy()
        group_labels[flip_mask] = 1 - group_labels[flip_mask]
    elif shortcut_type == "nonlinear":
        embeddings, labels = generate_nonlinear_shortcut(
            n_samples=n_samples,
            embedding_dim=n_features,
            seed=seed,
        )
        base = rng.standard_normal(size=embeddings.shape)
        embeddings = shortcut_strength * embeddings + (1.0 - shortcut_strength) * base
        flip_mask = rng.random(n_samples) > shortcut_strength
        group_labels = labels.copy()
        group_labels[flip_mask] = 1 - group_labels[flip_mask]
    elif shortcut_type == "none":
        embeddings, labels = generate_no_shortcut(
            n_samples=n_samples,
            embedding_dim=n_features,
            seed=seed,
        )
        nuisance_source = rng.standard_normal((n_samples, 1))
        nuisance_direction = rng.standard_normal((1, n_features))
        nuisance = nuisance_source @ nuisance_direction
        nuisance = nuisance / max(float(np.std(nuisance)), 1e-8)
        embeddings = embeddings + (0.5 * shortcut_strength * nuisance)
        group_labels = rng.integers(0, max(n_classes, 2), size=n_samples)
    else:
        return json.dumps(
            {
                "error": f"Unknown shortcut_type '{shortcut_type}'. Use 'linear', 'nonlinear', or 'none'."
            }
        )

    return json.dumps(
        {
            "embeddings": embeddings.tolist(),
            "labels": labels.tolist(),
            "group_labels": np.asarray(group_labels).tolist(),
            "description": (
                f"{shortcut_type} shortcut dataset: {n_samples} samples, "
                f"{n_features} features, strength={shortcut_strength}, seed={seed}"
            ),
        }
    )


def run_detector(
    embeddings: list[list[float]] | None = None,
    labels: list[int] | None = None,
    group_labels: list[int] | None = None,
    embeddings_path: str | None = None,
    labels_path: str | None = None,
    group_labels_path: str | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
    session_id: str | None = None,
) -> str:
    try:
        return _run_detector_impl(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            group_labels_path=group_labels_path,
            methods=methods,
            seed=seed,
            session_id=session_id,
            owner_key=None,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


def get_summary(session_id: str = "default") -> str:
    try:
        session = _load_session(session_id)
    except PermissionError:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    if session is None:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    detector = session.get("detector")
    if detector is not None:
        return detector.summary()
    return str(session["summary"])


def get_method_detail(method: str, session_id: str = "default") -> str:
    try:
        session = _load_session(session_id)
    except PermissionError:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    if session is None:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    raw = session["raw"]
    if method not in raw:
        available = list(raw.keys())
        return json.dumps({"error": f"Method '{method}' not found. Available: {available}"})
    return json.dumps(_safe_serialize(raw[method]), indent=2)


def compare_methods(
    embeddings: list[list[float]] | None = None,
    labels: list[int] | None = None,
    group_labels: list[int] | None = None,
    embeddings_path: str | None = None,
    labels_path: str | None = None,
    group_labels_path: str | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
) -> str:
    try:
        return _compare_methods_impl(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            group_labels_path=group_labels_path,
            methods=methods,
            seed=seed,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


def generate_report(
    session_id: str = "default",
    output_path: str | None = None,
    format: str = "html",
    include_visualizations: bool = True,
    export_csv: bool = False,
    csv_dir: str | None = None,
    return_base64: bool = False,
) -> str:
    try:
        return _generate_report_impl(
            session_id=session_id,
            output_path=output_path,
            format=format,
            include_visualizations=include_visualizations,
            export_csv=export_csv,
            csv_dir=csv_dir,
            return_base64=return_base64,
            owner_key=None,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


def run_benchmark(
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> str:
    try:
        return _run_benchmark_impl(config=config, config_path=config_path)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------


@mcp.tool(name="list_methods", description="List all available shortcut detection methods with short descriptions.")
def list_methods_tool() -> str:
    return list_methods()


@mcp.tool(name="generate_synthetic_data", description="Generate synthetic shortcut detection data with configurable type, strength, and size.")
def generate_synthetic_data_tool(
    n_samples: int = 200,
    n_features: int = 32,
    shortcut_strength: float = 0.8,
    shortcut_type: str = "linear",
    n_classes: int = 2,
    seed: int = 42,
) -> str:
    return generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        shortcut_strength=shortcut_strength,
        shortcut_type=shortcut_type,
        n_classes=n_classes,
        seed=seed,
    )


@mcp.tool(name="run_detector", description="Run one or more shortcut detection methods on embeddings and return a summary result.")
async def run_detector_tool(
    embeddings: list[list[float]] | None = None,
    labels: list[int] | None = None,
    group_labels: list[int] | None = None,
    embeddings_path: str | None = None,
    labels_path: str | None = None,
    group_labels_path: str | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    owner_key = _owner_key_from_ctx(ctx)
    try:
        return await _run_detector_async(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            group_labels_path=group_labels_path,
            methods=methods,
            seed=seed,
            session_id=session_id,
            owner_key=owner_key,
            ctx=ctx,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@mcp.tool(name="get_summary", description="Retrieve a human-readable summary for a previous shortcut detection session.")
async def get_summary_tool(
    session_id: str = "default",
    ctx: Context | None = None,
) -> str:
    try:
        session = await asyncio.to_thread(_load_session, session_id, _owner_key_from_ctx(ctx))
    except PermissionError:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    if session is None:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    detector = session.get("detector")
    if detector is not None:
        return detector.summary()
    return str(session["summary"])


@mcp.tool(name="get_method_detail", description="Get detailed raw results for a specific detection method from a stored session.")
async def get_method_detail_tool(
    method: str,
    session_id: str = "default",
    ctx: Context | None = None,
) -> str:
    try:
        session = await asyncio.to_thread(_load_session, session_id, _owner_key_from_ctx(ctx))
    except PermissionError:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    if session is None:
        return f"No results found for session_id='{session_id}'. Run run_detector() first."
    raw = session["raw"]
    if method not in raw:
        available = list(raw.keys())
        return json.dumps({"error": f"Method '{method}' not found. Available: {available}"})
    return json.dumps(_safe_serialize(raw[method]), indent=2)


@mcp.tool(name="compare_methods", description="Compare multiple shortcut detection methods side-by-side and return a consensus summary.")
async def compare_methods_tool(
    embeddings: list[list[float]] | None = None,
    labels: list[int] | None = None,
    group_labels: list[int] | None = None,
    embeddings_path: str | None = None,
    labels_path: str | None = None,
    group_labels_path: str | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
    ctx: Context | None = None,
) -> str:
    try:
        return await _compare_methods_async(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            group_labels_path=group_labels_path,
            methods=methods,
            seed=seed,
            ctx=ctx,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@mcp.tool(name="generate_report", description="Generate an HTML/PDF/Markdown report for a saved session and optionally return encoded content.")
async def generate_report_tool(
    session_id: str = "default",
    output_path: str | None = None,
    format: str = "html",
    include_visualizations: bool = True,
    export_csv: bool = False,
    csv_dir: str | None = None,
    return_base64: bool = False,
    ctx: Context | None = None,
) -> str:
    owner_key = _owner_key_from_ctx(ctx)
    try:
        if ctx is not None:
            await ctx.report_progress(0, 100, "Loading cached session")
        result = await asyncio.to_thread(
            _generate_report_impl,
            session_id=session_id,
            output_path=output_path,
            format=format,
            include_visualizations=include_visualizations,
            export_csv=export_csv,
            csv_dir=csv_dir,
            return_base64=return_base64,
            owner_key=owner_key,
        )
        if ctx is not None:
            await ctx.report_progress(100, 100, "Report generation complete")
        return result
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@mcp.tool(name="run_benchmark", description="Run the full synthetic benchmark suite or a provided benchmark configuration.")
async def run_benchmark_tool(
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    ctx: Context | None = None,
) -> str:
    try:
        if ctx is not None:
            await ctx.report_progress(0, 100, "Loading benchmark configuration")
        result = await asyncio.to_thread(
            _run_benchmark_impl,
            config=config,
            config_path=config_path,
        )
        if ctx is not None:
            await ctx.report_progress(100, 100, "Benchmark complete")
        return result
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
