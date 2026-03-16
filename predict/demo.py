import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from segearth_segmentor import SegEarthSegmentation


def build_parser():
    parser = argparse.ArgumentParser(
        description='SegEarth-OV demo: open-vocabulary segmentation for remote sensing images'
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
        '--clip-type',
        type=str,
        default='GeoRSCLIP',
        choices=['CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'],
        help='CLIP 模型类型（默认：GeoRSCLIP）'
    )
    parser.add_argument(
        '--vit-type',
        type=str,
        default='ViT-L-14',
        help='ViT 模型类型，如 ViT-B/16 或 ViT-L-14（默认：ViT-L-14，对应 GeoRSCLIP ViT-L-14 CLIPSelf 权重）'
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
        default='seg_pred.png',
        help='分割结果保存路径（默认：seg_pred.png）'
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
        # 如果用户没有提供现有的类别文件，回退到一个简单的 OpenEarthMap 风格类别列表
        name_list = [
            'background',
            'bareland,barren',
            'grass',
            'pavement',
            'road',
            'tree,forest',
            'water,river',
            'cropland',
            'building,roof,house',
        ]
        name_file = Path('./configs/my_name.txt')
        name_file.parent.mkdir(parents=True, exist_ok=True)
        with open(name_file, 'w') as writers:
            for i, name in enumerate(name_list):
                if i == len(name_list) - 1:
                    writers.write(name)
        else:
                    writers.write(name + '\n')

    # 图像预处理（与 OpenEarthMap 配置保持一致：448×448 + CLIP 标准归一化）
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]
            ),
            transforms.Resize((448, 448)),
    ])(img)

    img_tensor = img_tensor.unsqueeze(0).to(args.device)

    # 构建分割模型
    # 对于 GeoRSCLIP + ViT-L-14，会自动从 segearth_segmentor.py 中加载
    # checkpoint/RS5M_ViT-L-14_CLIPSelf.pth（已接入 MyExperiment 的 CLIPSelf 权重）
    # 注意：对于 768-d 特征（ViT-L-14），feature_up 会自动关闭（因为没有对应的 SimFeatUp 权重）
    # 但为了明确，我们也可以显式设置 feature_up=False
    model = SegEarthSegmentation(
                clip_type=args.clip_type,      # 默认：GeoRSCLIP
                vit_type=args.vit_type,        # 默认：ViT-L-14
                model_type='SegEarth',         # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
                name_path=str(name_file),
                feature_up=False,              # 显式关闭 feature_up（避免 CUDA 扩展问题，且 768-d 没有对应权重）
    )

    seg_pred = model.predict(img_tensor, data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

            # 可视化与保存
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(seg_pred, cmap='viridis')
    ax[1].axis('off')
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f'保存分割结果到: {out_path}')


if __name__ == '__main__':
    main()
