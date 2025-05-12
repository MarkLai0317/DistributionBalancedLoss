#!/bin/sh

# Number of groups passed as argument
N=$1

# Template
TEMPLATE_FILE=configs/voc/15_2/template.py

# Base paths
DATA_ROOT="/home/mark/Desktop/工研院/multi-label_classification/data/voc"
ONLINE_BASE="${DATA_ROOT}/35_6_group"

# Output directory
OUTPUT_DIR="configs/voc/35_6"

if [ -z "$N" ]; then
  echo "Usage: sh generate_voc_configs.sh <number_of_groups>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for i in $(seq 1 "$N")
do
  echo "Generating VOC config for group $i..."

  CSV_PATH="${DATA_ROOT}/group3_l${i}_r0_train_data.csv"
  NUM_CLASSES=$(head -n 1 "$CSV_PATH" | awk -F',' '{print NF-1}')
  ONLINE_PATH="${DATA_ROOT}/35_6_group3_l${i}_r0/"
  WORK_DIR="/media/mark/T7\ Shield/work_dirs/LT_voc_resnet50_pfc_DB_uniform_bce_35_6_group3_l${i}_r0"
  OUTPUT_FILE="${OUTPUT_DIR}/LT_resnet50_pfc_DB_uniform_bce_35_6_group3_l${i}_r0.py"

  sed \
    -e "s/^\s*num_classes\s*=.*/        num_classes=${NUM_CLASSES},/" \
    -e "s|^\s*online_data_root\s*=.*|online_data_root = \"${ONLINE_PATH}\"|" \
    -e "s|^\s*work_dir\s*=.*|work_dir = \"${WORK_DIR}\"|" \
    "$TEMPLATE_FILE" > "$OUTPUT_FILE"
done
