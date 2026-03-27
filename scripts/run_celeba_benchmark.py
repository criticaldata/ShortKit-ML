#!/usr/bin/env python3
"""Run shortcut detection benchmark on CelebA data.

Loads CelebA embeddings + metadata and runs ShortcutDetector with multiple
methods on known shortcut pairs:
  - Blond_Hair prediction with Male as sensitive attribute
  - Heavy_Makeup prediction with Male as sensitive attribute

Saves results to output/paper_benchmark/celeba_results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so shortcut_detect is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shortcut_detect.benchmark.method_utils import ALL_METHODS  # noqa: E402
from shortcut_detect.unified import ShortcutDetector  # noqa: E402

DATA_DIR = ROOT / "data" / "celeba"
OUTPUT_DIR = ROOT / "output" / "paper_benchmark" / "celeba_results"

# Shortcut pairs: (task_attribute, sensitive_attribute, description)
SHORTCUT_PAIRS = [
    ("Blond_Hair", "Male", "Blond hair strongly correlated with gender"),
    ("Heavy_Makeup", "Male", "Heavy makeup strongly correlated with gender"),
]

# Methods to run
METHODS = list(ALL_METHODS)


def load_celeba_data(data_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load embeddings, metadata, and raw attributes."""
    emb_path = data_dir / "celeba_embeddings.npy"
    meta_path = data_dir / "celeba_metadata.csv"
    attr_path = data_dir / "celeba_attributes.csv"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_path}. Run setup_celeba_data.py first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. Run setup_celeba_data.py first."
        )

    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)

    if attr_path.exists():
        attributes = pd.read_csv(attr_path)
    else:
        # Reconstruct from metadata attr_* columns
        attr_cols = [c for c in metadata.columns if c.startswith("attr_")]
        attributes = metadata[["image_id"] + attr_cols].copy()
        attributes.columns = ["image_id"] + [c.replace("attr_", "") for c in attr_cols]

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded metadata: {len(metadata)} rows")
    print(f"Loaded attributes: {len(attributes)} rows, {len(attributes.columns)-1} attributes")
    return embeddings, metadata, attributes


def run_shortcut_pair(
    embeddings: np.ndarray,
    attributes: pd.DataFrame,
    task_attr: str,
    sensitive_attr: str,
    methods: list[str],
    seed: int = 42,
) -> dict:
    """Run shortcut detection for one task/sensitive attribute pair."""
    task_labels = attributes[task_attr].values.astype(int)
    group_labels = attributes[sensitive_attr].values.astype(int)

    print(f"\n{'='*70}")
    print(f"Task: {task_attr} | Sensitive: {sensitive_attr}")
    print(
        f"  Task label distribution: {dict(zip(*np.unique(task_labels, return_counts=True), strict=False))}"
    )
    print(
        f"  Group label distribution: {dict(zip(*np.unique(group_labels, return_counts=True), strict=False))}"
    )

    # Cross-tabulation
    ct = pd.crosstab(
        pd.Series(task_labels, name=task_attr),
        pd.Series(group_labels, name=sensitive_attr),
    )
    print(f"  Cross-tabulation:\n{ct}\n")

    detector = ShortcutDetector(methods=methods, seed=seed)
    detector.fit(embeddings, task_labels, group_labels=group_labels)

    summary = detector.summary()
    print(summary)

    results = detector.get_results()
    return {
        "task_attr": task_attr,
        "sensitive_attr": sensitive_attr,
        "summary": summary,
        "results": _serialize_results(results),
        "shortcut_detected": _check_shortcut_detected(results),
    }


def _serialize_results(results: dict) -> dict:
    """Convert results dict to JSON-serializable form."""
    serialized = {}
    for method, res in results.items():
        ser = {}
        for k, v in res.items():
            if k == "detector":
                continue
            if isinstance(v, np.ndarray):
                ser[k] = v.tolist()
            elif isinstance(v, np.floating):
                ser[k] = float(v)
            elif isinstance(v, np.integer):
                ser[k] = int(v)
            elif isinstance(v, dict):
                # Recursively handle nested dicts (e.g., by_attribute)
                ser[k] = _serialize_nested(v)
            else:
                try:
                    json.dumps(v)
                    ser[k] = v
                except (TypeError, ValueError):
                    ser[k] = str(v)
        serialized[method] = ser
    return serialized


def _serialize_nested(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k == "detector":
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.floating | np.float64):
            out[k] = float(v)
        elif isinstance(v, np.integer | np.int64):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = _serialize_nested(v)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _check_shortcut_detected(results: dict) -> dict:
    """Check per-method whether a shortcut was detected."""
    detected = {}
    for method, res in results.items():
        # Try several possible key names for the risk assessment
        risk = res.get("risk_label") or res.get("risk_level") or res.get("risk") or "unknown"
        if isinstance(risk, str):
            detected[method] = risk.lower() in ("high", "moderate")
        else:
            detected[method] = bool(risk)
    return detected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CelebA shortcut detection benchmark")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        help="Detection methods to use",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    embeddings, metadata, attributes = load_celeba_data(args.data_dir)

    all_results = []
    for task_attr, sensitive_attr, description in SHORTCUT_PAIRS:
        if task_attr not in attributes.columns:
            print(f"WARNING: {task_attr} not in attributes, skipping.")
            continue
        if sensitive_attr not in attributes.columns:
            print(f"WARNING: {sensitive_attr} not in attributes, skipping.")
            continue

        pair_result = run_shortcut_pair(
            embeddings=embeddings,
            attributes=attributes,
            task_attr=task_attr,
            sensitive_attr=sensitive_attr,
            methods=args.methods,
            seed=args.seed,
        )
        pair_result["description"] = description
        all_results.append(pair_result)

        # Save individual pair results
        pair_file = args.output_dir / f"{task_attr}_vs_{sensitive_attr}.json"
        with open(pair_file, "w") as f:
            json.dump(pair_result, f, indent=2, default=str)
        print(f"\nSaved: {pair_file}")

        # Save summary text
        summary_file = args.output_dir / f"{task_attr}_vs_{sensitive_attr}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(pair_result["summary"])

    # Aggregate report
    print("\n" + "=" * 70)
    print("CELEBA BENCHMARK AGGREGATE REPORT")
    print("=" * 70)

    report_lines = []
    for r in all_results:
        task = r["task_attr"]
        sens = r["sensitive_attr"]
        detected = r["shortcut_detected"]
        line = f"{task} vs {sens}: {r['description']}"
        report_lines.append(line)
        for method, found in detected.items():
            status = "DETECTED" if found else "NOT DETECTED"
            report_lines.append(f"  {method}: {status}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save aggregate
    agg_path = args.output_dir / "aggregate_report.json"
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
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved aggregate report: {agg_path}")

    report_txt_path = args.output_dir / "aggregate_report.txt"
    with open(report_txt_path, "w") as f:
        f.write(report_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
