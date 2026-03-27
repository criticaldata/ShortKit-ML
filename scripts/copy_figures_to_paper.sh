#!/bin/bash
# Copy all generated paper figures to the Overleaf paper figures/ directory.
# Run after generate_paper_figures.py, generate_umap_plots.py, generate_correction_plot.py.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
SRC="$REPO_ROOT/output/paper_figures"
DST="$REPO_ROOT/docs-temp/shortcut-paper/figures"

mkdir -p "$DST"

if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory not found: $SRC"
    echo "Run the figure generation scripts first."
    exit 1
fi

count=0
for f in "$SRC"/*.pdf "$SRC"/*.png; do
    [ -f "$f" ] || continue
    cp "$f" "$DST/"
    echo "  Copied: $(basename "$f")"
    count=$((count + 1))
done

echo "Done. Copied $count files to $DST"
