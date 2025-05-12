#!/bin/sh

MAX_PARALLEL=3
START=1
END=10
COUNTER=0

for i in $(seq $START $END)
do
  echo "Launching training for group2 step$i..."
  python tools/train.py configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group2_step${i}.py &

  COUNTER=$((COUNTER+1))

  # Wait if MAX_PARALLEL jobs are running
  if [ "$COUNTER" -ge "$MAX_PARALLEL" ]; then
    wait
    COUNTER=0
  fi
done

# Wait for any remaining background jobs to finish
wait
