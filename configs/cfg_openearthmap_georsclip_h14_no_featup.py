_base_ = './base_config.py'

# GeoRSCLIP ViT-H-14 在 OpenEarthMap 上的评估配置（不使用 SimFeatUp）
# 说明：
# - 使用 SegEarth-OV 框架（SegEarthSegmentation）
# - backbone: GeoRSCLIP ViT-H-14
# - feature_up=False：显式关闭 SimFeatUp，上采样仅使用双线性插值

model = dict(
    clip_type='GeoRSCLIP',
    vit_type='ViT-H-14',
    model_type='SegEarth',   # 保持与原框架一致
    feature_up=False,        # 关闭 SimFeatUp
    name_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1,
)

# dataset settings
dataset_type = 'OpenEarthMapDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='data/OpenEarthMap/img_dir/val',
            seg_map_path='data/OpenEarthMap/ann_dir/val'),
        pipeline=test_pipeline))



