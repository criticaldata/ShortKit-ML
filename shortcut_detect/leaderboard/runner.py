"""Score a submission on its track and write ``results.json``."""

from __future__ import annotations

import json
import platform
import time

import numpy as np

from . import embedding_tracks, image_tracks
from .loader import Submission, submission_imports

SCORERS = {
    "embedding-detection": embedding_tracks.score_detection,
    "embedding-mitigation": embedding_tracks.score_mitigation,
    "image-detection": image_tracks.score_detection,
    "image-mitigation": image_tracks.score_mitigation,
}


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def score_submission(sub: Submission, suite: str = "full", seed_offset: int = 0) -> dict:
    t0 = time.perf_counter()
    with submission_imports(sub.path):
        out = SCORERS[sub.metadata["track"]](sub.entry, suite=suite, seed_offset=seed_offset)
    from .. import __version__

    return _clean(
        {
            "method": sub.name,
            "track": sub.metadata["track"],
            "benchmark_version": sub.track.version,
            "suite": suite,
            "seed_offset": seed_offset,
            "primary_metric": sub.track.primary,
            "metrics": out["metrics"],
            "seconds_per_dataset": out["seconds_per_dataset"],
            "total_seconds": time.perf_counter() - t0,
            "environment": {"shortkit_ml": __version__, "python": platform.python_version()},
            "details": out["details"],
        }
    )


def write_results(sub: Submission, results: dict) -> None:
    sub.results_path.write_text(json.dumps(results, indent=2) + "\n")


def compare(old: dict, new: dict, tolerance: float) -> list[str]:
    """Differences between committed and recomputed metrics beyond ``tolerance``."""
    problems = []
    if old.get("benchmark_version") != new.get("benchmark_version"):
        problems.append(
            f"results.json was scored on benchmark {old.get('benchmark_version')}, current is {new['benchmark_version']}"
        )
    old_metrics = old.get("metrics", {})
    for k in sorted(set(old_metrics) - set(new["metrics"])):
        problems.append(f"{k}: committed {old_metrics[k]}, no longer produced")
    for k, v in new["metrics"].items():
        o = old_metrics.get(k)
        if o is None or v is None:
            if o != v:
                problems.append(f"{k}: committed {o}, recomputed {v}")
        elif abs(o - v) > tolerance:
            problems.append(f"{k}: committed {o:.4f}, recomputed {v:.4f} (tolerance {tolerance})")
    return problems
