#!/usr/bin/env bash
# End-to-end revision pipeline (CMPB-D-26-02542). Called by
# `scripts/reproduce_paper.sh revision`; can also run selected stages:
#   scripts/revision/run_all.sh [stage ...]
# Stages: old clinical negctl synthetic nullcheck head analyses umap marker extraction alignment tables (default: all, in order)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python3}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -z "${SHORTKIT_REVISION_DIR:-}" ]; then
    export SHORTKIT_REVISION_DIR="${HOME}/Documents/shortkit-ml-results/2026-10-28_revision_$(git -C "${REPO}" rev-parse --short HEAD)"
fi
mkdir -p "${SHORTKIT_REVISION_DIR}"
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(old clinical negctl synthetic nullcheck head analyses umap marker extraction alignment tables)
echo "results: ${SHORTKIT_REVISION_DIR}"
for s in "${STAGES[@]}"; do
    start=$(date +%s)
    echo ">>> stage ${s} ($(date))"
    case "${s}" in
        old)       "${PY}" "${REPO}/scripts/revision/run_old_pipeline.py" ;;
        clinical)  "${PY}" "${REPO}/scripts/revision/run_clinical_audit.py" ;;
        negctl)    "${PY}" "${REPO}/scripts/revision/run_negative_control.py" ;;
        synthetic) "${PY}" "${REPO}/scripts/revision/run_synthetic.py" ;;
        nullcheck) "${PY}" "${REPO}/scripts/revision/run_null_check.py" ;;
        head)      "${PY}" "${REPO}/scripts/revision/run_head_experiment.py" ;;
        analyses)  "${PY}" "${REPO}/scripts/revision/run_analyses.py" ;;
        umap)      "${PY}" "${REPO}/scripts/revision/run_umap.py" ;;
        marker)    "${PY}" "${REPO}/scripts/revision/run_planted_marker.py" ;;
        # needs torch >= 2.6 with transformers (EXTRACT_PYTHON) and the MPS lease
        extraction) "${EXTRACT_PYTHON:-${PY}}" "${REPO}/scripts/revision/extract_missing_chexpert_backbones.py" --device "${EXTRACT_DEVICE:-cpu}" ;;
        alignment) "${PY}" "${REPO}/scripts/revision/run_alignment_check.py" ;;
        tables)    "${PY}" "${REPO}/scripts/revision/make_tables_figures.py" ;;
        *) echo "unknown stage ${s}" >&2; exit 1 ;;
    esac
    echo "<<< stage ${s} done in $(( $(date +%s) - start ))s"
done
