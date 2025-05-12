# model settings
model = dict(
    type='SimpleClassifier',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        type='PretrainResNet50',
        ),
    neck=dict(
        type='PFC',
        in_channels=2048,
        out_channels=256,
        dropout=0.5),
    head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=15,
        method='fc',
        loss_cls=dict(
            type='BCELoss', reduction='mean')))
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'VOCDataset'
data_root = '/home/mark/Desktop/工研院/multi-label_classification/data/voc/'
online_data_root = "/home/mark/Desktop/工研院/multi-label_classification/data/voc/35_6_group3_l7_r0/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    sampler='RandomSampler',
    train=dict(
            type=dataset_type,
            ann_file=online_data_root + 'voc_annotations.pkl',
            img_prefix=data_root + 'train_images',
            img_scale=(224, 224),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5
    ),
    val=dict(
        type=dataset_type,
        ann_file=online_data_root + 'voc_annotations_test.pkl',
        img_prefix=data_root + 'test_images',
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0),
    test=dict(
        type=dataset_type,
        ann_file=online_data_root + 'voc_annotations_test.pkl',
        img_prefix=data_root + 'test_images',
        class_split=online_data_root + 'class_split.pkl',
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[55, 75])
checkpoint_config = dict(interval=1, create_symlink=False)
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=5)
# runtime settings
total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = "/media/mark/T7 Shield/work_dirs/LT_voc_resnet50_pfc_DB_uniform_bce_35_6_group3_l7_r0"
load_from = None
resume_from = None
workflow = [('train', 1)]
