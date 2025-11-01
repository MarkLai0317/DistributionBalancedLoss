#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
MODE=${4:-"test"}  # Default to test mode if not specified

PORT=${MASTER_PORT:-29500}

# Generate prediction file name based on checkpoint and mode
PRED_FILE="${CHECKPOINT%.*}_predictions_${MODE}.pkl"
# Generate JSON output file name
JSON_FILE="${CHECKPOINT%.*}_predictions_${MODE}_evaluation.json"

echo "Step 1/2: Running distributed prediction on ${MODE} dataset..."
$PYTHON -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    "$(dirname "$0")/predict.py" "$CONFIG" "$CHECKPOINT" --mode "$MODE" --out "$PRED_FILE" "${@:5}"

# Check if prediction succeeded
if [[ $? -eq 0 && -f "$PRED_FILE" ]]; then
    echo "Step 2/2: Evaluating predictions and generating JSON results..."
    $PYTHON "$(dirname "$0")/evaluate.py" "$CONFIG" "$PRED_FILE" --out-json "$JSON_FILE" "${@:5}"
    
else
    echo "Prediction failed or prediction file not found!"
    exit 1
fi
