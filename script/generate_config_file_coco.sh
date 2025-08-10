#!/bin/sh

# Number of groups passed as argument
N=$1

# Base config template
TEMPLATE_FILE=configs/coco/15_2/config_template.py

# Absolute data path base
DATA_ROOT="/home/mark/Desktop/工研院/multi-label_classification/data/coco"
# ONLINE_BASE="${DATA_ROOT}/35_6_group1_l0_r"

# Output directory
OUTPUT_DIR="configs/coco/35_6_CVIR"

# Validate input
if [ -z "$N" ]; then
  echo "Usage: sh generate_configs.sh <number_of_groups>"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

for i in $(seq 1 "$N")
do
  echo "Generating config for group $i..."

  CSV_PATH="${DATA_ROOT}/group2_step${i}_train_data.csv"
  NUM_CLASSES=$(head -n 1 "$CSV_PATH" | awk -F',' '{print NF-1}')
  ONLINE_PATH="${DATA_ROOT}/35_6_CVIR_group2_step${i}/"
  WORK_DIR="./work_dirs/LT_coco_resnet50_pfc_DB_uniform_bce_35_6_CVIR_group2_step${i}"
  OUTPUT_FILE="${OUTPUT_DIR}/LT_resnet50_pfc_DB_uniform_bce_35_6_CVIR_group2_step${i}.py"

  sed \
    -e "s/^\s*num_classes\s*=.*/        num_classes=${NUM_CLASSES},/" \
    -e "s|^\s*online_data_root\s*=.*|online_data_root = \"${ONLINE_PATH}\"|" \
    -e "s|^\s*work_dir\s*=.*|work_dir = \"${WORK_DIR}\"|" \
    "$TEMPLATE_FILE" > "$OUTPUT_FILE"
done
