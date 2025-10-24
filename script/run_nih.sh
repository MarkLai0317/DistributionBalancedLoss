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
echo "Creating long-tail dataset..."
python tools/create_longtail_dataset.py

if [ ! -f "$LONGTAILDATAROOT/img_id.txt" ]; then
    echo "❌ Error: img_id.txt not created. Check create_longtail_dataset.py"
    exit 1
fi

echo "✅ Long-tail dataset created at $LONGTAILDATAROOT"

# ============================================================================
# Step 3: Organize Images
# ============================================================================
echo ""
echo "========================================"
echo "Step 3: Organize Images"
echo "========================================"
echo "Moving images into train/ and test/ directories..."

python tools/split_nih_images.py \
  --image-root "$DATAROOT" \
  --train-txt "$LONGTAILDATAROOT/img_id.txt" \
  --test-txt "$DATAROOT/test_list.txt" \
  --output-root "$DATAROOT"

echo "✅ Images organized"

# ============================================================================
# Step 4: Convert Dataset Format
# ============================================================================
echo ""
echo "========================================"
echo "Step 4: Convert Dataset Format"
echo "========================================"
echo "Converting to model-compatible format..."

python tools/df_to_csv_and_pkl.py \
  --df "$DATAROOT/Data_Entry_2017.csv" \
  --image-root "$DATAROOT" \
  --train-txt "$LONGTAILDATAROOT/img_id.txt" \
  --test-txt "$DATAROOT/test_list.txt" \
  --output-dir "$LONGTAILDATAROOT"

# Check if all required files are generated
REQUIRED_FILES=(
    "$LONGTAILDATAROOT/train_annotations.pkl"
    "$LONGTAILDATAROOT/train_data.csv"
    "$LONGTAILDATAROOT/test_annotations.pkl"
    "$LONGTAILDATAROOT/test_data.csv"
    "$LONGTAILDATAROOT/class_freq.pkl"
)

echo "Checking generated files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found"
        exit 1
    else
        echo "✅ $file"
    fi
done

# ============================================================================
# Step 5: Start Training
# ============================================================================
echo ""
echo "========================================"
echo "Step 5: Start Training"
echo "========================================"
echo "Starting training on GPU $CUDA_DEVICE..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python tools/train.py configs/nih/LT_resnet50_pfc_DB.py --validate

# ============================================================================
# Step 6: Testing (Optional)
# ============================================================================
echo ""
echo "========================================"
echo "Step 6: Testing"
echo "========================================"
bash tools/dist_test.sh configs/nih/LT_resnet50_pfc_DB.py work_dirs/LT_coco_resnet50_pfc_DB_pretrain_test/epoch_80.pth 1
echo ""
echo "✅ All steps completed!"
