#!/bin/sh

MAX_PARALLEL=3
START=1
END=14
COUNTER=0

for i in $(seq $START $END)
do
  echo "Launching training for group1 l0 r$i..."
  python tools/train.py configs/voc/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group1_l0_r${i}.py &

  COUNTER=$((COUNTER+1))

  # Wait if MAX_PARALLEL jobs are running
  if [ "$COUNTER" -ge "$MAX_PARALLEL" ]; then
    wait
    COUNTER=0
  fi
done

# Wait for any remaining background jobs to finish
wait
