#!/bin/bash

# Multi-Epoch Evaluation and Plotting Script
# This script evaluates models at different epochs and generates metric plots

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

CONFIG="configs/nih/LT_resnet50_pfc_DB.py"
WORK_DIR="work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test"

# GPU device
GPUS=1

# Dataset mode: train, val, or test (default: test)
MODE=${1:-"test"}

# Epoch range: 5, 10, 15, 20, ..., 80
EPOCHS=($(seq 10 10 80))

echo "========================================"
echo "Multi-Epoch Evaluation Script"
echo "========================================"
echo "Config: $CONFIG"
echo "Work dir: $WORK_DIR"
echo "Mode: $MODE"
echo "Epochs to evaluate: ${EPOCHS[@]}"
echo ""

# ============================================================================
# Step 2: Evaluate each epoch
# ============================================================================
echo ""
echo "========================================"
echo "Step 2: Evaluating epochs on $MODE dataset"
echo "========================================"

for epoch in "${EPOCHS[@]}"; do
    checkpoint="$WORK_DIR/epoch_${epoch}.pth"
    
    echo ""
    echo "----------------------------------------"
    echo "Evaluating epoch $epoch on $MODE dataset"
    echo "Checkpoint: $checkpoint"
    echo "----------------------------------------"
    
    # Run evaluation and capture output
    echo "Running evaluation..."
    bash tools/dist_test.sh "$CONFIG" "$checkpoint" "$GPUS" "$MODE"
    
    echo "✅ Epoch $epoch completed"
done

echo ""
echo "✅ All evaluations completed!"