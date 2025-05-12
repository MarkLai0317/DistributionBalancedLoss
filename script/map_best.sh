#!/bin/bash
group_number=$1
# CONFIG="configs/voc/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group${group_number}.py"
# DIR="./work_dirs/LT_voc_resnet50_pfc_DB_2_uniform_bce_35_6_group${group_number}"
# CONFIG="configs/coco/LT_resnet50_pfc_DB_uniform_focal_all.py"
# DIR="work_dirs/LT_coco_resnet50_pfc_DB_uniform_focal_all_test"
# CONFIG="configs/coco/15_2/LT_resnet50_pfc_DB_uniform_bce_15_2_group${group_number}.py"
# DIR="work_dirs/LT_coco_resnet50_pfc_DB_uniform_bce_15_2_group${group_number}"
# CONFIG="configs/voc/15_2/LT_resnet50_pfc_DB_uniform_bce_15_2_group${group_number}.py"
# DIR="work_dirs/LT_voc_resnet50_pfc_DB_uniform_bce_15_2_group${group_number}"
# CONFIG="configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group1_l0_r${group_number}.py"
# DIR="work_dirs/LT_coco_resnet50_pfc_DB_uniform_bce_35_6_group1_l0_r${group_number}"
# CONFIG="configs/coco/split/LT_resnet50_pfc_DB_uniform_bce_split_group${group_number}.py"
# DIR="work_dirs/LT_coco_resnet50_pfc_DB_uniform_bce_split_group${group_number}"

# CONFIG="configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group2_step${group_number}.py"
# DIR="work_dirs/LT_coco_resnet50_pfc_DB_uniform_bce_35_6_group2_step${group_number}"

CONFIG="configs/voc/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group2_step${group_number}.py"
DIR="work_dirs/LT_voc_resnet50_pfc_DB_uniform_bce_35_6_group2_step${group_number}"


BEST_EPOCH_MACRO=1
BEST_SCORE_MACRO=0.0
BEST_EPOCH_MAP=1
BEST_SCORE_MAP=0.0

BEST_TOTAL_EPOCH_MACRO=1
BEST_TOTAL_SCORE_MACRO=0.0
BEST_TOTAL_EPOCH_MAP=1
BEST_TOTAL_SCORE_MAP=0.0

for epoch in $(seq 1 1 80)
do
    echo "Evaluating epoch $epoch"
    OUTPUT=$(bash tools/dist_test.sh $CONFIG "$DIR/epoch_${epoch}.pth" 1)

    # Extract scores from mid split
    MID_MAP=$(echo "$OUTPUT" | grep -Eo "Split:\s+mid.*mAP:[0-9.]+" | grep -Eo "mAP:[0-9.]+" | cut -d':' -f2)
    MID_MACRO=$(echo "$OUTPUT" | grep -Eo "Split:\s+mid.*macro:[0-9.]+" | grep -Eo "macro:[0-9.]+" | cut -d':' -f2)

    # Extract scores from mid split
    total_MAP=$(echo "$OUTPUT" | grep -Eo "Split:\s+Total.*mAP:[0-9.]+" | grep -Eo "mAP:[0-9.]+" | cut -d':' -f2)
    total_MACRO=$(echo "$OUTPUT" | grep -Eo "Split:\s+Total.*macro:[0-9.]+" | grep -Eo "macro:[0-9.]+" | cut -d':' -f2)


    # Update best mid macro score
    if (( $(echo "$total_MACRO > $BEST_TOTAL_SCORE_MACRO" | bc -l) )); then
        BEST_TOTAL_SCORE_MACRO=$total_MACRO
        BEST_TOTAL_EPOCH_MACRO=$epoch
    fi

    # Update best mid mAP score
    if (( $(echo "$total_MAP > $BEST_TOTAL_SCORE_MAP" | bc -l) )); then
        BEST_TOTAL_SCORE_MAP=$total_MAP
        BEST_TOTAL_EPOCH_MAP=$epoch
    fi
    
    if (( $(echo "$MID_MACRO > $BEST_SCORE_MACRO" | bc -l) )); then
        BEST_SCORE_MACRO=$MID_MACRO
        BEST_EPOCH_MACRO=$epoch
    fi

    # Update best mid mAP score
    if (( $(echo "$MID_MAP > $BEST_SCORE_MAP" | bc -l) )); then
        BEST_SCORE_MAP=$MID_MAP
        BEST_EPOCH_MAP=$epoch
    fi
    echo "Best epoch for mid macro score: $BEST_EPOCH_MACRO with score: $BEST_SCORE_MACRO"
    echo "Best epoch for mid mAP: $BEST_EPOCH_MAP with score: $BEST_SCORE_MAP"

    echo "Best epoch for total macro score: $BEST_TOTAL_EPOCH_MACRO with score: $BEST_TOTAL_SCORE_MACRO"
    echo "Best epoch for total mAP: $BEST_TOTAL_EPOCH_MAP with score: $BEST_TOTAL_SCORE_MAP"
done

echo "Best epoch for mid macro score: $BEST_EPOCH_MACRO with score: $BEST_SCORE_MACRO"
echo "Best epoch for mid mAP: $BEST_EPOCH_MAP with score: $BEST_SCORE_MAP"

echo "Best epoch for total macro score: $BEST_TOTAL_EPOCH_MACRO with score: $BEST_TOTAL_SCORE_MACRO"
# echo "Best epoch for total mAP: $BEST_TOTAL_EPOCH_MAP with score: $BEST_TOTAL_SCORE_MAP" > "voc/35_6_group3/voc_uniform_bce_35_6_group3_l${group_number}_r0.txt"
echo "Best epoch for total mAP: $BEST_TOTAL_EPOCH_MAP with score: $BEST_TOTAL_SCORE_MAP" > "voc/voc_uniform_bce_35_6_group2_step${group_number}.txt"
