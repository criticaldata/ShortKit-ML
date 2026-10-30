#!/usr/bin/env bash
# End-to-end revision pipeline (CMPB-D-26-02542). Called by
# `scripts/reproduce_paper.sh revision`; can also run selected stages:
#   scripts/revision/run_all.sh [stage ...]
# Stages: old clinical negctl exchange synthetic nullcheck head dfr mlp race analyses umap marker extraction alignment overlap sensitivity sample_vs_full nlprobe nlprobe_clinical scaling tables (default: all, in order)
# Run on request only: prepare_mimic (rebuilds two MIMIC inputs from the coreset parquet files in the hub dataset)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python3}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -z "${SHORTKIT_REVISION_DIR:-}" ]; then
    export SHORTKIT_REVISION_DIR="${HOME}/Documents/shortkit-ml-results/2026-10-28_revision_$(git -C "${REPO}" rev-parse --short HEAD)"
fi
mkdir -p "${SHORTKIT_REVISION_DIR}"
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(old clinical negctl exchange synthetic nullcheck head dfr mlp race analyses umap marker extraction alignment overlap sensitivity sample_vs_full nlprobe nlprobe_clinical scaling tables)
echo "results: ${SHORTKIT_REVISION_DIR}"
for s in "${STAGES[@]}"; do
    start=$(date +%s)
    echo ">>> stage ${s} ($(date))"
    case "${s}" in
        # rebuilds data/mimic_cxr/vit32_cls_embeddings_realigned.npy and mimic_race_raw.csv; needs pyarrow; the parquet files come from the hub dataset (mimic_cxr/source/) or QML_DIR
        prepare_mimic) "${PY}" "${REPO}/scripts/revision/prepare_mimic_source.py" ${QML_DIR:+--qml "${QML_DIR}"} ;;
        old)       "${PY}" "${REPO}/scripts/revision/run_old_pipeline.py" ;;
        clinical)  "${PY}" "${REPO}/scripts/revision/run_clinical_audit.py" ;;
        negctl)    "${PY}" "${REPO}/scripts/revision/run_negative_control.py" ;;
        synthetic) "${PY}" "${REPO}/scripts/revision/run_synthetic.py" ;;
        nullcheck) "${PY}" "${REPO}/scripts/revision/run_null_check.py" ;;
        exchange)  "${PY}" "${REPO}/scripts/revision/run_patient_exchangeability.py" ;;
        head)      "${PY}" "${REPO}/scripts/revision/run_head_experiment.py" ;;
        dfr)       "${PY}" "${REPO}/scripts/revision/run_dfr_control.py" ;;
        mlp)       "${PY}" "${REPO}/scripts/revision/run_head_mlp.py" ;;
        race)      "${PY}" "${REPO}/scripts/revision/run_head_race.py" ;;
        analyses)  "${PY}" "${REPO}/scripts/revision/run_analyses.py" ;;
        umap)      "${PY}" "${REPO}/scripts/revision/run_umap.py" ;;
        marker)    "${PY}" "${REPO}/scripts/revision/run_planted_marker.py" ;;
        # needs torch >= 2.6 with transformers (EXTRACT_PYTHON) and the MPS lease
        extraction) "${EXTRACT_PYTHON:-${PY}}" "${REPO}/scripts/revision/extract_missing_chexpert_backbones.py" --device "${EXTRACT_DEVICE:-cpu}" ;;
        alignment) "${PY}" "${REPO}/scripts/revision/run_alignment_check.py" ;;
        overlap)   "${PY}" "${REPO}/scripts/revision/run_pretraining_overlap.py" ;;
        sensitivity) for c in C frontal labelled; do "${PY}" "${REPO}/scripts/revision/run_probe_sensitivity.py" --check "${c}"; done ;;
        sample_vs_full) "${PY}" "${REPO}/scripts/revision/run_sample_vs_full.py" ;;
        nlprobe)   "${PY}" "${REPO}/scripts/revision/run_nonlinear_probe_check.py" ;;
        # squared-feature probe on the 33 primary clinical cells, K = 500; about 10 min with 3 jobs
        nlprobe_clinical) "${PY}" "${REPO}/scripts/revision/run_nonlinear_probe_clinical.py" ;;
        # probe run time against n on synthetic embeddings (d = 768); about 15 min on a laptop CPU
        scaling)   "${PY}" "${REPO}/scripts/revision/run_probe_scaling.py" ;;
        # clinical_audit_stored_arrays/ (the "Stored array" column of the sensitivity table) has
        # no stage: it is the clinical stage's output from before the CheXpert switch to public
        # re-extractions (8728cb3), kept under that name as a frozen input. The stored arrays have
        # no extraction code, so it cannot be regenerated from this pipeline.
        tables)    "${PY}" "${REPO}/scripts/revision/make_tables_figures.py" ;;
        *) echo "unknown stage ${s}" >&2; exit 1 ;;
    esac
    echo "<<< stage ${s} done in $(( $(date +%s) - start ))s"
done
