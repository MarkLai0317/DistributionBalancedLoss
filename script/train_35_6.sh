# (
# python tools/train.py configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group1.py
# ) &
# (
# python tools/train.py configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group2.py
# ) &
# (
# python tools/train.py configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group3.py
# ) 
# wait

# (
# python tools/train.py configs/voc/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group1.py
# ) &
# (
# python tools/train.py configs/voc/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group2.py
# ) &
# (
# python tools/train.py configs/voc/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group3.py
# ) 

(
python tools/train.py configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group1_2.py
) &
(
python tools/train.py configs/coco/35_6/LT_resnet50_pfc_DB_uniform_bce_35_6_group2_3.py
)
wait
