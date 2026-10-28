"""Write numbers.json as LaTeX macros so the paper never types a number by hand.

Each key becomes ``\\csname num@<key>\\endcsname``; the manuscript's ``\\NUM{key}``
looks it up and prints a red placeholder only when the key is missing.

Usage:
    python scripts/revision/export_numbers_tex.py output/paper_tables/revision/numbers.json \
        output/paper_tables/revision/tables/numbers.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LATEX_SPECIAL = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_"}


def to_latex(value: object) -> str:
    text = str(value)
    for char, escaped in LATEX_SPECIAL.items():
        text = text.replace(char, escaped)
    return text


def main(src: str, dst: str) -> int:
    numbers = json.loads(Path(src).read_text())
    lines = [f"% Generated from {Path(src).name} by {Path(__file__).name}. Do not edit."]
    for key in sorted(numbers):
        value = numbers[key]
        if key == "raw" or isinstance(value, (dict, list)):
            continue
        lines.append(f"\\expandafter\\def\\csname num@{key}\\endcsname{{{to_latex(value)}}}")
    Path(dst).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} macros to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
