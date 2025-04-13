#!/bin/bash
# group_number=$1
# CONFIG="configs/voc/LT_resnet50_pfc_DB_no_extra_group${group_number}.py"
# DIR="work_dirs/LT_voc_resnet50_pfc_DB_no_extra_group${group_number}"
CONFIG="configs/voc/LT_resnet50_pfc_DB.py"
DIR="work_dirs/LT_voc_resnet50_pfc_DB_classaware_bce"

BEST_EPOCH_MACRO=1
BEST_SCORE_MACRO=0.0
BEST_EPOCH_MAP=1
BEST_SCORE_MAP=0.0

BEST_TOTAL_EPOCH_MACRO=1
BEST_TOTAL_SCORE_MACRO=0.0
BEST_TOTAL_EPOCH_MAP=1
BEST_TOTAL_SCORE_MAP=0.0

for epoch in $(seq 1 8)
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
echo "Best epoch for total mAP: $BEST_TOTAL_EPOCH_MAP with score: $BEST_TOTAL_SCORE_MAP" > "voc_new_classaware_DB.txt"
