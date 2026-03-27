#!/usr/bin/env python3
"""Run shortcut detection benchmark on REAL CelebA embeddings.

Loads real CelebA embeddings (extracted from actual images via ResNet-50)
and runs ShortcutDetector on known shortcut pairs:
  - Blond_Hair prediction with Male as sensitive attribute
  - Heavy_Makeup prediction with Male as sensitive attribute
  - Attractive prediction with Male as sensitive attribute

Saves results to output/paper_benchmark/celeba_real_results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shortcut_detect.benchmark.method_utils import ALL_METHODS  # noqa: E402
from shortcut_detect.unified import ShortcutDetector  # noqa: E402

DATA_DIR = ROOT / "data" / "celeba"
OUTPUT_DIR = ROOT / "output" / "paper_benchmark" / "celeba_real_results"

SHORTCUT_PAIRS = [
    ("Blond_Hair", "Male", "Blond hair strongly correlated with gender"),
    ("Heavy_Makeup", "Male", "Heavy makeup strongly correlated with gender"),
    ("Attractive", "Male", "Attractiveness label correlated with gender"),
]

METHODS = list(ALL_METHODS)


def load_data() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    emb_path = DATA_DIR / "celeba_real_embeddings.npy"
    meta_path = DATA_DIR / "celeba_real_metadata.csv"
    attr_path = DATA_DIR / "celeba_real_attributes.csv"

    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    attributes = pd.read_csv(attr_path)

    print(f"Embeddings: {embeddings.shape}")
    print(f"Metadata: {len(metadata)} rows")
    print(f"Attributes: {attributes.shape}")
    return embeddings, metadata, attributes


def _serialize(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating | np.float64 | np.float32):
        return float(obj)
    if isinstance(obj, np.integer | np.int64 | np.int32):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items() if k != "detector"}
    if isinstance(obj, list | tuple):
        return [_serialize(x) for x in obj]
    return obj


def run_pair(embeddings, attributes, task_attr, sensitive_attr, description):
    task_labels = attributes[task_attr].values.astype(int)
    group_labels = attributes[sensitive_attr].values.astype(int)

    print(f"\n{'='*70}")
    print(f"Task: {task_attr} | Sensitive: {sensitive_attr}")
    print(f"  {description}")

    ct = pd.crosstab(
        pd.Series(task_labels, name=task_attr),
        pd.Series(group_labels, name=sensitive_attr),
    )
    print(f"  Cross-tabulation:\n{ct}\n")

    # Build extra_labels for intersectional analysis
    # Include numeric attribute columns (CelebA binary attrs are 0/1 ints).
    extra_labels: dict[str, np.ndarray] = {}
    for col in attributes.select_dtypes(include="number").columns:
        if col != task_attr:
            extra_labels[col] = attributes[col].values.astype(int)

    detector = ShortcutDetector(methods=METHODS, seed=42)
    detector.fit(embeddings, task_labels, group_labels=group_labels, extra_labels=extra_labels)

    summary = detector.summary()
    print(summary)

    results = detector.get_results()
    serialized = _serialize(results)

    # Check detection
    detected = {}
    for method, res in results.items():
        risk = res.get("risk_label") or res.get("risk_level") or res.get("risk") or "unknown"
        if isinstance(risk, str):
            detected[method] = risk.lower() in ("high", "moderate")
        else:
            detected[method] = bool(risk)

    return {
        "task_attr": task_attr,
        "sensitive_attr": sensitive_attr,
        "description": description,
        "summary": summary,
        "results": serialized,
        "shortcut_detected": detected,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, metadata, attributes = load_data()

    all_results = []
    for task_attr, sensitive_attr, description in SHORTCUT_PAIRS:
        result = run_pair(embeddings, attributes, task_attr, sensitive_attr, description)
        all_results.append(result)

        pair_file = OUTPUT_DIR / f"{task_attr}_vs_{sensitive_attr}.json"
        with open(pair_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved: {pair_file}")

        summary_file = OUTPUT_DIR / f"{task_attr}_vs_{sensitive_attr}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(result["summary"])

    # Aggregate report
    print(f"\n{'='*70}")
    print("REAL CELEBA BENCHMARK — AGGREGATE REPORT")
    print("=" * 70)

    lines = []
    for r in all_results:
        lines.append(f"\n{r['task_attr']} vs {r['sensitive_attr']}: {r['description']}")
        for method, found in r["shortcut_detected"].items():
            status = "DETECTED" if found else "NOT DETECTED"
            lines.append(f"  {method}: {status}")

    report = "\n".join(lines)
    print(report)

    agg_path = OUTPUT_DIR / "aggregate_report.json"
    with open(agg_path, "w") as f:
        json.dump(
            {
                "shortcut_pairs": [
                    {
                        "task": r["task_attr"],
                        "sensitive": r["sensitive_attr"],
                        "description": r["description"],
                        "detected_by": r["shortcut_detected"],
                    }
                    for r in all_results
                ]
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {agg_path}")

    report_path = OUTPUT_DIR / "aggregate_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
