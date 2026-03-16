import os
import sys
import argparse

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# 把项目根目录加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # 项目根目录，即SegEarth-OV目录
if ROOT not in sys.path: # 如果项目根目录不在 sys.path 中，则加入 sys.path
    sys.path.insert(0, ROOT) # 将项目根目录加入 sys.path

import segearth_segmentor  # noqa: F401
import custom_datasets  # noqa: F401

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate GeoRSCLIP(ViT-B-32 finetuned visual) on OpenEarthMap (SegEarth-OV)')
    p.add_argument(
        '--config',
        default='./configs/cfg_openearthmap_reorsclip_vitb32_no_featup_no_cls_sub.py'
    )
    p.add_argument('--work-dir', default='./work_logs/openearthmap_reorsclip_vitb32/')
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

    # override checkpoint paths from CLI
    cfg.model.visual_pretrained_path = args.visual_ckpt
    cfg.model.text_pretrained_path = args.text_ckpt

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()

