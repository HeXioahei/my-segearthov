# python eval/eval_openearthmap_reorsclip_vitb32.py \
#   --work-dir ./work_logs/openearthmap_georsclip_vitb32_finetuned_visual \
#   --visual-ckpt /root/checkpoint/best_model.pth \
#   --text-ckpt /root/checkpoint/RS5M_ViT-B-32.pt

# python eval_openearthmap_reorsclip_vitb32.py \
#   --work-dir ./work_logs/loveda_georsclip_vitb32_finetuned_visual \
#   --visual-ckpt /root/checkpoint/best_model.pth \
#   --text-ckpt /root/checkpoint/RS5M_ViT-B-32.pt

# python eval_openearthmap_selfdistill_vitb32.py \
#   --work-dir ./work_logs/openearthmap_georsclip_vitb32_selfdistill_forward \
#   --visual-ckpt /root/checkpoint/best_model.pth \
#   --text-ckpt /root/checkpoint/RS5M_ViT-B-32.pt

python eval/eval.py \
  --config ./configs/cfg_openearthmap_selfdistill_featmap_vitb32.py \
  --work-dir ./work_logs/openearthmap_georsclip_vitb32_selfdistill_featmap