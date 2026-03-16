import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from segearth_segmentor import SegEarthSegmentation


def build_parser():
    parser = argparse.ArgumentParser(
        description='SegEarth-OV demo (no featup): 原始 GeoRSCLIP ViT-L-14 open-vocabulary segmentation'
    )
    parser.add_argument(
        '--img',
        type=str,
        default='demo/oem_koeln_50.tif',
        help='输入图像路径（默认：demo/oem_koeln_50.tif）'
    )
    parser.add_argument(
        '--name-file',
        type=str,
        default='./configs/cls_openearthmap.txt',
        help='类别名称文件，每行一个/多个同义词，用逗号分隔（默认：configs/cls_openearthmap.txt）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='推理设备（默认：cuda）'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='seg_pred_georsclip_l14_original_no_featup.png',
        help='分割结果保存路径（默认：seg_pred_georsclip_l14_original_no_featup.png）'
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.is_file():
        raise FileNotFoundError(f'输入图像不存在: {img_path}')

    # 读取图像
    img = Image.open(img_path)

    # 读取 / 构造类别名称文件
    name_file = Path(args.name_file)
    if not name_file.is_file():
        raise FileNotFoundError(f'类别名称文件不存在: {name_file}')

    # 图像预处理：与 CLIP / SegEarth-OV 保持一致
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]
        ),
        transforms.Resize((448, 448)),
    ])(img)

    img_tensor = img_tensor.unsqueeze(0).to(args.device)

    # 构建原始 GeoRSCLIP ViT-L-14 模型（无 featup）：
    #   - clip_type='GeoRSCLIP'
    #   - vit_type='ViT-L-14'
    #   - feature_up=False（显式关闭 SimFeatUp，因为 768-d 特征没有对应的 upsampler 权重）
    print('[demo_georsclip_l_no_featup] 使用原始 GeoRSCLIP ViT-L-14 权重（未经过 CLIPSelf 训练）')
    print('[demo_georsclip_l_no_featup] feature_up 已关闭（768-d 特征无对应的 SimFeatUp 权重）')
    
    model = SegEarthSegmentation(
        clip_type='GeoRSCLIP',
        vit_type='ViT-B-32',
        model_type='SegEarth',
        name_path=str(name_file),
        feature_up=False,  # 显式关闭，因为 768-d 没有对应的 upsampler 权重
    )

    seg_pred = model.predict(img_tensor, data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

    # 可视化与保存
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original Image', fontsize=12)
    ax[1].imshow(seg_pred, cmap='viridis')
    ax[1].axis('off')
    ax[1].set_title('Segmentation (GeoRSCLIP ViT-B-32)', fontsize=12)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'保存分割结果到: {out_path}')


if __name__ == '__main__':
    main()

