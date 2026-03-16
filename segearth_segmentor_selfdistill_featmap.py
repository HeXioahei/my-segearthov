import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

from prompts.imagenet_template import *
from SelfDistill.src.open_clip import create_model, tokenizer


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices


@MODELS.register_module()
class SegEarthSegmentationSelfDistillFeatMap(BaseSegmentor):
    """使用 SelfDistill/test.py 的特征图提取逻辑做 OVSS 分割."""

    def __init__(
        self,
        name_path: str,
        device: torch.device = torch.device('cuda'),
        prob_thd: float = 0.0,
        logit_scale: float = 50.0,
        slide_stride: int = 112,
        slide_crop: int = 224,
        bg_idx: int = 0,
        visual_pretrained_path: str = '/root/checkpoint/best_model.pth',
        text_pretrained_path: str = '/root/checkpoint/RS5M_ViT-B-32.pt',
        **kwargs,
    ):
        # 与 SegEarth-OV 一致的数据预处理
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)

        self.device = device
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.logit_scale = logit_scale
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        # 1) 创建完整 CLIP 模型（视觉+文本来自 RS5M）
        self.net = create_model(
            'ViT-B/32',
            pretrained=text_pretrained_path,
            precision='fp32',
            device=device,
        )
        self.net.eval()

        # 2) 只加载学生视觉编码器权重（best_model.pth）
        self._load_visual_only(self.net, visual_pretrained_path)

        # 3) 构建文本原型（与 SegEarth-OV 一致）
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.tensor(self.query_idx, dtype=torch.int64, device=device)

        tok = tokenizer.tokenize
        query_features = []
        with torch.no_grad():
            for qw in query_words:
                texts = [temp(qw) for temp in openai_imagenet_template]
                text_tokens = tok(texts).to(device)
                feat = self.net.encode_text(text_tokens)  # [T, dim]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat = feat.mean(dim=0)
                feat = feat / feat.norm()
                query_features.append(feat.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)  # [Nq, dim]
        self.patch_size = self.net.visual.patch_size
        self.dtype = self.query_features.dtype

    @staticmethod
    def _load_visual_only(clip_model: nn.Module, ckpt_path: Optional[str]):
        if not ckpt_path:
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
        if not isinstance(state, dict):
            raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')
        if state and next(iter(state.keys())).startswith('module.'):
            state = {k[7:]: v for k, v in state.items()}

        # 如果 state 是完整 CLIP，则抽出 visual.*
        if any(k.startswith('visual.') for k in state.keys()):
            state = {k[len('visual.'):]: v for k, v in state.items() if k.startswith('visual.')}

        incompatible = clip_model.visual.load_state_dict(state, strict=False)
        print(f'[SelfDistillFeatMap] loaded visual from {ckpt_path}, '
              f'missing={len(incompatible.missing_keys)}, '
              f'unexpected={len(incompatible.unexpected_keys)}',
              flush=True)

    def forward_feature(self, img, logit_size=None):
        # 参考 SelfDistill/test.py：visual(x) -> (pooled, feature_map)
        if isinstance(img, list):
            img = img[0]
        img = img.to(self.device).float()  # test.py 用的是 fp32

        with torch.no_grad():
            pooled, feat_map = self.net.visual(img)  # [B,512], [B,grid^2,768] 或 [B,D,Hf,Wf] 取决于实现

        # 如果 feat_map 是 [B,N,D]，reshape 成 [B,D,Hf,Wf]
        if feat_map.dim() == 3:
            B, N, D = feat_map.shape
            Hf = Wf = int(N ** 0.5)
            feat_map = feat_map.view(B, Hf, Wf, D).permute(0, 3, 1, 2)  # [B,D,Hf,Wf]

        B, Dv, Hf, Wf = feat_map.shape  # Dv 通常为 768

        # 投影到 embedding 维度（512），与文本空间一致
        if getattr(self.net.visual, 'proj', None) is not None:
            # [B,Dv,Hf,Wf] -> [B,Hf,Wf,Dv] @ [Dv,512] -> [B,Hf,Wf,512] -> [B,512,Hf,Wf]
            feat_map_proj = feat_map.permute(0, 2, 3, 1) @ self.net.visual.proj
            feat_map_proj = feat_map_proj.permute(0, 3, 1, 2).contiguous()
        else:
            # 退化：直接当作已经是 512 维
            feat_map_proj = feat_map

        # 展平并归一化： [B,512,Hf,Wf] -> [B,Hf*Wf,512]
        B, C, Hf, Wf = feat_map_proj.shape
        feat_flat = feat_map_proj.view(B, C, Hf * Wf).permute(0, 2, 1)  # [B,N,512]
        feat_flat = feat_flat / feat_flat.norm(dim=-1, keepdim=True)

        text_feats = self.query_features.to(feat_flat.device).to(feat_flat.dtype)  # [Nq,512]

        logits = feat_flat @ text_feats.T  # [B,N,Nq]
        logits = logits.permute(0, 2, 1).reshape(B, self.num_queries, Hf, Wf)  # [B,Nq,Hf,Wf]

        if logit_size is None:
            logits = F.interpolate(logits, size=img.shape[-2:], mode='bilinear', align_corners=False)
        else:
            logits = F.interpolate(logits, size=logit_size, mode='bilinear', align_corners=False)

        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        if isinstance(img, list):
            img = img[0].unsqueeze(0)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])
                if any(pad):
                    crop_img = F.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img)

                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += F.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = F.interpolate(preds, size=img_size, mode='bilinear', align_corners=False)
        return logits

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0]
            )] * inputs.shape[0]
        # 保持与 test.py 一致，用 fp32
        if isinstance(inputs, list):
            x = inputs[0]
        else:
            x = inputs
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(x, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(x, batch_img_metas[0]['ori_shape'])
        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        B = seg_logits.shape[0]
        for i in range(B):
            seg_logits_i = seg_logits[i] * self.logit_scale
            seg_logits_i = seg_logits_i.softmax(0)

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits_i = seg_logits_i.unsqueeze(0)
                cls_index = F.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits_i = (seg_logits_i * cls_index).max(1)[0]

            seg_pred = seg_logits_i.argmax(0, keepdim=True)
            seg_pred[seg_logits_i.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits': PixelData(**{'data': seg_logits_i}),
                    'pred_sem_seg': PixelData(**{'data': seg_pred}),
                })
        return data_samples

    @staticmethod
    def compute_padsize(H: int, W: int, patch_size: int):
        l = r = t = b = 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l
        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t
        return l, r, t, b

    # 训练相关接口占位
    def loss(self, inputs, data_samples, **kwargs):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError

    def _forward(self, inputs, data_samples=None, **kwargs):
        return self.predict(inputs, data_samples)