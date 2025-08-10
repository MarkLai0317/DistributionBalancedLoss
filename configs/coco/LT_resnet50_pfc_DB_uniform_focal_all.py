# model settings
model = dict(
    type='SimpleClassifier',
    backbone=dict(
        type='PretrainResNet50',),
    neck=dict(
        type='PFC',
        in_channels=2048,
        out_channels=256,
        dropout=0),
    head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=80,
        method='fc',
        loss_cls=dict(
          type='FocalLoss', reduction='mean')))
            # type='ResampleLoss', use_sigmoid=True,
            # reweight_func='rebalance',
            # focal=dict(focal=True, balance_param=2.0, gamma=2),
            # logit_reg=dict(neg_scale=2.0, init_bias=0.05),
            # map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
            # loss_weight=1.0, freq_file='appendix/coco/longtail2017/class_freq.pkl')))
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/mark/Desktop/工研院/multi-label_classification/data/coco/'
online_data_root = '/home/mark/Desktop/工研院/multi-label_classification/data/coco/DB/'
img_norm_cfg = dict(to_rgb=True)
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

img_size=224
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    sampler='RandomSampler',
    train=dict(
            type=dataset_type,
            ann_file=online_data_root + 'coco_annotations.pkl',
            img_prefix=data_root + 'train_images',
            img_scale=(img_size, img_size),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5),
    val=dict(
        type=dataset_type,
        ann_file=online_data_root + 'coco_annotations_test.pkl',
        img_prefix=data_root + 'test_images',
        img_scale=(img_size, img_size),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0),
    test=dict(
        type=dataset_type,
        ann_file=online_data_root + 'coco_annotations_test.pkl',
        img_prefix=data_root + 'test_images',
        class_split=online_data_root + 'class_split.pkl',
        img_scale=(img_size, img_size),
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
    step=[55,75])  # 8: [5,7]) 4: [2,3]) 40: [25,35]) 80: [55,75])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=59,
    hooks=[
        dict(type='TextLoggerHook'),
    ])


# # optimizer = dict(type='Adam', lr=0.002)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[55,75])  # 8: [5,7]) 4: [2,3]) 40: [25,35]) 80: [55,75])
# lr_config = dict(
#     policy='OneCycle',
#     max_lr=0.002,
#     total_steps=705 * 8,
#     div_factor=25,
#     final_div_factor=100
# )


# yapf:enable
evaluation = dict(interval=5)
# runtime settings
start_epoch=0
total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/LT_coco_resnet50_pfc_DB_uniform_focal_all2_test'
load_from = None
if start_epoch > 0:
    resume_from = work_dir + '/epoch_{}.pth'.format(start_epoch)
    print("start from epoch {}".format(start_epoch))
else:
    resume_from = None
workflow = [('train', 1)]