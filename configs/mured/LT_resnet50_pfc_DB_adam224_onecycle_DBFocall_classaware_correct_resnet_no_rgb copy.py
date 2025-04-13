# model settings
model = dict(
    type='SimpleClassifier',
    backbone=dict(
        type='PretrainResNet50',
        ),
    neck=dict(
        type='PFC',
        in_channels=2048,
        out_channels=256,
        dropout=0),
    head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=20,
        method='fc',
        loss_cls=dict(
            type='ResampleLoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(neg_scale=2.0, init_bias=0.05),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
            loss_weight=1.0, freq_file='appendix/mured/class_freq.pkl')))
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'MuredDataset'
data_root = 'appendix/mured/'
online_data_root = 'appendix/'
img_norm_cfg = dict(
    )
extra_aug = dict(
    # photo_metric_distortion=dict(
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18
    # ),
    # random_crop=dict(
    #     min_crop_size=0.8
    # )
)

img_size=224
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    sampler='ClassAwareSampler',
    train=dict(
            type=dataset_type,
            ann_file=data_root + 'mured_annotations.pkl',
            img_prefix=data_root + 'images/',
            img_scale=(img_size, img_size),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'mured_annotations_test.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(img_size, img_size),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'mured_annotations_test.pkl',
        class_split=online_data_root + 'mured/class_split.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(img_size, img_size),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32, 
        resize_keep_ratio=False,
        flip_ratio=0))
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.000075)
optimizer_config = dict()
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[5,7])  # 8: [5,7]) 4: [2,3]) 40: [25,35]) 80: [55,75])
lr_config = dict(
    policy='OneCycle',
    max_lr=0.000075,
    total_steps=61 * 80,
    div_factor=25,
    final_div_factor=100
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=61,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=5)
# runtime settings
start_epoch=0
total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/LT_mured_resnet50_pfc_DB_adam224_onecycle_DBFocal_classaware_resnet50_no_rgb'
load_from = None
if start_epoch > 0:
    resume_from = work_dir + '/epoch_{}.pth'.format(start_epoch)
    print("start from epoch {}".format(start_epoch))
else:
    resume_from = None
workflow = [('train', 1)]