#!/bin/bash
# Extract CheXpert embeddings from 3 backbones in parallel.
# First downloads/caches the data, then runs 3 backbone extractions.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DEVICE="${1:-mps}"
N_SAMPLES="${2:-2000}"

echo "=== Extracting CheXpert embeddings: $N_SAMPLES samples, device=$DEVICE ==="

# Run each backbone in parallel
python3 scripts/extract_chexpert_hf_multibackbone.py \
    --n-samples "$N_SAMPLES" --device "$DEVICE" --batch-size 32 \
    --backbones resnet50 &
PID1=$!

python3 scripts/extract_chexpert_hf_multibackbone.py \
    --n-samples "$N_SAMPLES" --device "$DEVICE" --batch-size 32 \
    --backbones densenet121 &
PID2=$!

python3 scripts/extract_chexpert_hf_multibackbone.py \
    --n-samples "$N_SAMPLES" --device "$DEVICE" --batch-size 32 \
    --backbones vit_b_16 &
PID3=$!

echo "Launched 3 processes: resnet50=$PID1 densenet121=$PID2 vit_b_16=$PID3"
echo "Waiting for all to complete..."

wait $PID1 && echo "resnet50 done" || echo "resnet50 FAILED"
wait $PID2 && echo "densenet121 done" || echo "densenet121 FAILED"
wait $PID3 && echo "vit_b_16 done" || echo "vit_b_16 FAILED"

echo "=== All done ==="
ls -lh data/chexpert_multibackbone/
