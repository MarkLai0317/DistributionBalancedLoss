#!/bin/bash

# NIH Dataset Training Pipeline
# This script automates the steps described in run_nih_guide.md

set -e  # Exit on error

# ============================================================================
# Configuration - UPDATE THESE PATHS
# ============================================================================

# Original dataset root (where you downloaded the NIH dataset)
DATAROOT="../NIH_dataset"

# Long-tail resampled dataset root (will be created)
LONGTAILDATAROOT="./appendix/nih/longtailnih"

# GPU device
CUDA_DEVICE=4


# ============================================================================
# Step 2: Resample the Dataset
# ============================================================================
echo ""
echo "========================================"
echo "Step 2: Resample the Dataset"
echo "========================================"
# echo "Creating long-tail dataset..."
# python tools/create_longtail_dataset.py

# if [ ! -f "$LONGTAILDATAROOT/img_id.txt" ]; then
#     echo "❌ Error: img_id.txt not created. Check create_longtail_dataset.py"
#     exit 1
# fi

# echo "✅ Long-tail dataset created at $LONGTAILDATAROOT"

# ============================================================================
# Step 3: Organize Images
# ============================================================================
echo ""
echo "========================================"
echo "Step 3: Organize Images"
echo "========================================"
# echo "Moving images into train/ and test/ directories..."

# python tools/split_nih_images.py \
#   --image-root "$DATAROOT" \
#   --train-txt "$LONGTAILDATAROOT/img_id.txt" \
#   --test-txt "$DATAROOT/test_list.txt" \
#   --output-root "$DATAROOT"

# echo "✅ Images organized"

# ============================================================================
# Step 3.5: Split Train/Eval Sets
# ============================================================================
echo ""
echo "========================================"
echo "Step 3.5: Split Train/Eval Sets"
echo "========================================"
# echo "Splitting training data into train (85%) and eval (15%) sets..."

# python tools/train_eval_split.py \
#   --input "$LONGTAILDATAROOT/img_id.txt" \
#   --output-dir "$LONGTAILDATAROOT" \
#   --train-ratio 0.85 \
#   --seed 42

# echo "✅ Train/eval split completed"
# echo "  - Train set: $LONGTAILDATAROOT/train_img_id.txt"
# echo "  - Eval set: $LONGTAILDATAROOT/eval_img_id.txt"

# ============================================================================
# Step 4: Convert Dataset Format
# ============================================================================
echo ""
echo "========================================"
echo "Step 4: Convert Dataset Format"
echo "========================================"
# echo "Converting to model-compatible format..."

# python tools/df_to_csv_and_pkl.py \
#   --df "$DATAROOT/Data_Entry_2017.csv" \
#   --image-root "$DATAROOT" \
#   --train-txt "$LONGTAILDATAROOT/train_img_id.txt" \
#   --eval-txt "$LONGTAILDATAROOT/eval_img_id.txt" \
#   --test-txt "$DATAROOT/test_list.txt" \
#   --output-dir "$LONGTAILDATAROOT"

# echo "✅ Dataset format conversion completed"
# echo "  - Train annotations: $LONGTAILDATAROOT/train_annotations.pkl"
# echo "  - Eval annotations: $LONGTAILDATAROOT/eval_annotations.pkl"
# echo "  - Test annotations: $LONGTAILDATAROOT/test_annotations.pkl"

# # ============================================================================
# # Step 5: Start Training
# # ============================================================================
echo ""
echo "========================================"
echo "Step 5: Start Training"
echo "========================================"
echo "Starting training on GPU $CUDA_DEVICE..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python tools/train.py configs/nih/LT_resnet50_pfc_DB.py

# # ============================================================================
# # Step 6: Testing (Optional)
# # ============================================================================
# echo ""
# echo "========================================"
# echo "Step 6: Testing"
# echo "========================================"
# bash tools/dist_test.sh configs/nih/LT_resnet50_pfc_DB.py work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test/epoch_80.pth 1
# echo ""
# echo "✅ All steps completed!"
