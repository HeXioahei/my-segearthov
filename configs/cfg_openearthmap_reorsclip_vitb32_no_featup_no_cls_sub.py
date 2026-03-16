_base_ = './base_config.py'

# GeoRSCLIP (ViT-B-32, finetuned VISUAL tower) on OpenEarthMap (val)
# - visual: finetuned GeoRSCLIP visual encoder (SelfDistill student) -> visual_pretrained_path
# - text: keep original RS5M weights -> text_pretrained_path
# - disable SimFeatUp: feature_up=False
# - remove CLS suppression ("subtract cls token"): cls_token_lambda=0

model = dict(
    clip_type='GeoRSCLIP',
    vit_type='ViT-B-32',
    model_type='SegEarth',
    feature_up=False,
    cls_token_lambda=0,
    name_path='./configs/cls_openearthmap.txt',
    # name_path='./configs/cls_loveda.txt',
    prob_thd=0.1,
    visual_pretrained_path='/root/checkpoint/best_model.pth',
    text_pretrained_path='/root/checkpoint/RS5M_ViT-B-32.pt',
)

# dataset settings
dataset_type = 'OpenEarthMapDataset'
# dataset_type = 'LoveDADataset'
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
            # img_path='data/LoveDA/img_dir/val',
            # seg_map_path='data/LoveDA/ann_dir/val'),
        pipeline=test_pipeline))

