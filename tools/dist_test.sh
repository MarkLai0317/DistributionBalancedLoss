#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

PORT=${MASTER_PORT:-29500}

# Generate prediction file name based on checkpoint
PRED_FILE="${CHECKPOINT%.*}_predictions.pkl"

echo "Step 1/2: Running distributed prediction..."
$PYTHON -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    "$(dirname "$0")/predict.py" "$CONFIG" "$CHECKPOINT" --launcher none --out "$PRED_FILE" "${@:4}"

# Check if prediction succeeded
if [[ $? -eq 0 && -f "$PRED_FILE" ]]; then
    echo "Step 2/2: Evaluating predictions..."
    $PYTHON "$(dirname "$0")/evaluate.py" "$CONFIG" "$PRED_FILE" "${@:4}"
else
    echo "Prediction failed or prediction file not found!"
    exit 1
fi
