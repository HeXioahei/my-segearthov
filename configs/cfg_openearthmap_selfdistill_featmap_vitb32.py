_base_ = './base_config.py'

model = dict(
    type='SegEarthSegmentationSelfDistillFeatMap',
    name_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1,
    # 文本塔 + 初始视觉：原始 RS5M 权重
    text_pretrained_path='/root/checkpoint/RS5M_ViT-B-32.pt',
    # 学生视觉塔：SelfDistill 训练得到的 best_model
    visual_pretrained_path='/root/checkpoint/best_model.pth',
)

dataset_type = 'OpenEarthMapDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
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