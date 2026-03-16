import os
import argparse

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import segearth_segmentor_selfdistill  # noqa: F401
import custom_datasets  # noqa: F401

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate GeoRSCLIP(ViT-B-32) with SelfDistill forward on OpenEarthMap')
    p.add_argument(
        '--config',
        default='./configs/cfg_openearthmap_selfdistill_vitb32.py'
    )
    p.add_argument('--work-dir', default='./work_logs/openearthmap_georsclip_vitb32_selfdistill_forward/')
    p.add_argument('--visual-ckpt', default='/root/checkpoint/best_model.pth')
    p.add_argument('--text-ckpt', default='/root/checkpoint/RS5M_ViT-B-32.pt')
    p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = p.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir

    # 从命令行覆盖权重路径
    cfg.model.visual_pretrained_path = args.visual_ckpt
    cfg.model.text_pretrained_path = args.text_ckpt

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()

