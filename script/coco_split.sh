(
python tools/train.py configs/coco/split/LT_resnet50_pfc_DB_uniform_bce_split_group1.py 
) &
(
python tools/train.py configs/coco/split/LT_resnet50_pfc_DB_uniform_bce_split_group2.py 
) &
(
python tools/train.py configs/coco/split/LT_resnet50_pfc_DB_uniform_bce_split_group3.py
)
