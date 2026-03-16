_base_ = './base_config.py'

# 使用 SelfDistill 的 forward 逻辑评估 GeoRSCLIP ViT-B-32（只替换视觉塔）

model = dict(
    type='SegEarthSegmentationSelfDistill',
    name_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1, # 概率阈值
    # RS5M 原始 GeoRSCLIP 权重（完整 CLIP，用作文本塔 + 视觉初始化）
    text_pretrained_path='/root/checkpoint/RS5M_ViT-B-32.pt',
    # SelfDistill 训练后的视觉塔权重（student）
    visual_pretrained_path='/root/checkpoint/best_model.pth',
)

dataset_type = 'OpenEarthMapDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
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

