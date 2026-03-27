#!/usr/bin/env bash
# Reproduce all paper benchmark results for ShortKIT-ML.
#
# Usage:
#   ./scripts/reproduce_paper.sh [smoke|default|full]
#
# Profiles:
#   smoke   - Quick sanity check (~2-5 min). Minimal grid, 2 seeds.
#   default - Moderate run (~15-30 min). Reduced grid, 3+ seeds.
#   full    - Complete paper reproduction (~2-4 hours). Full grid, 10 seeds.
#
# The script is idempotent: re-running it creates a new timestamped output
# directory, leaving previous results untouched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-default}"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/output/paper_benchmark_${PROFILE}_${TIMESTAMP}"

# ---------------------------------------------------------------------------
# Validate profile
# ---------------------------------------------------------------------------
case "${PROFILE}" in
    smoke|default|full) ;;
    *)
        echo "ERROR: Unknown profile '${PROFILE}'. Choose: smoke, default, full" >&2
        exit 1
        ;;
esac

echo "============================================================"
echo " ShortKIT-ML Paper Benchmark Reproduction"
echo "============================================================"
echo " Profile   : ${PROFILE}"
echo " Output    : ${OUTPUT_DIR}"
echo " Timestamp : ${TIMESTAMP}"
echo " Python    : $(python3 --version 2>&1)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Select config
# ---------------------------------------------------------------------------
if [ "${PROFILE}" = "full" ]; then
    CONFIG="${REPO_ROOT}/examples/paper_benchmark_config_reproducible.json"
else
    # For smoke and default, generate a temporary config with the right profile
    CONFIG=$(mktemp "${TMPDIR:-/tmp}/shortcut_bench_XXXXXX.json")
    trap 'rm -f "${CONFIG}"' EXIT
    python3 -c "
import json, pathlib
cfg = json.loads(pathlib.Path('${REPO_ROOT}/examples/paper_benchmark_config_reproducible.json').read_text())
cfg['profile'] = '${PROFILE}'
cfg['output_dir'] = '${OUTPUT_DIR}'
# Remove comment keys for cleanliness
cfg = {k: v for k, v in cfg.items() if not k.startswith('_comment')}
if 'synthetic' in cfg:
    cfg['synthetic'] = {k: v for k, v in cfg['synthetic'].items() if not k.startswith('_comment')}
pathlib.Path('${CONFIG}').write_text(json.dumps(cfg, indent=2))
"
fi

# For full profile, also patch the output dir into the config
if [ "${PROFILE}" = "full" ]; then
    FULL_CONFIG=$(mktemp "${TMPDIR:-/tmp}/shortcut_bench_XXXXXX.json")
    trap 'rm -f "${FULL_CONFIG}"' EXIT
    python3 -c "
import json, pathlib
cfg = json.loads(pathlib.Path('${CONFIG}').read_text())
cfg['output_dir'] = '${OUTPUT_DIR}'
cfg = {k: v for k, v in cfg.items() if not k.startswith('_comment')}
if 'synthetic' in cfg:
    cfg['synthetic'] = {k: v for k, v in cfg['synthetic'].items() if not k.startswith('_comment')}
pathlib.Path('${FULL_CONFIG}').write_text(json.dumps(cfg, indent=2))
"
    CONFIG="${FULL_CONFIG}"
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Run synthetic benchmarks (Dataset 1)
# ---------------------------------------------------------------------------
START_TIME=$(date +%s)

echo ""
echo ">>> Running synthetic benchmarks (Dataset 1) ..."
echo ""

python3 -m shortcut_detect.benchmark.paper_runner --config "${CONFIG}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Benchmark Complete"
echo "============================================================"
echo " Profile      : ${PROFILE}"
echo " Elapsed time : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo " Output dir   : ${OUTPUT_DIR}"
echo ""
echo " Output files:"
if [ -d "${OUTPUT_DIR}" ]; then
    find "${OUTPUT_DIR}" -type f | sort | while read -r f; do
        SIZE=$(du -h "$f" | cut -f1)
        echo "   ${SIZE}  ${f#${OUTPUT_DIR}/}"
    done
else
    echo "   (no output directory found)"
fi
echo "============================================================"
