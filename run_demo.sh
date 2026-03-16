# python demo.py \
#   --img demo/oem_koeln_50.tif \
#   --name-file ./configs/cls_openearthmap.txt \
#   --clip-type GeoRSCLIP \
#   --vit-type ViT-L-14 \
#   --out seg_pred_clipself_vitl14.png

# python demo_clip_no_featup.py \
#   --img demo/oem_koeln_50.tif \
#   --name-file ./configs/cls_openearthmap.txt \
#   --device cuda \
#   --out seg_pred_clip_b16_no_featup.png

# python demo_georsclip_l_no_featup.py \
#   --img demo/oem_koeln_50.tif \
#   --name-file ./configs/cls_openearthmap.txt \
#   --device cuda \
#   --out seg_pred_georsclip_l14_original_no_featup.png

# python demo.py \
#   --img demo/oem_koeln_50.tif \
#   --name-file ./configs/cls_openearthmap.txt \
#   --clip-type GeoRSCLIP \
#   --vit-type ViT-B/32 \
#   --out seg_pred_GeoRSCLIP_vitb32.png

# python demo_georsclip_b_selfdistill_no_featup.py \
#   --img demo/oem_koeln_50.tif \
#   --name-file ./configs/cls_openearthmap.txt \
#   --device cuda \
#   --visual-ckpt /root/checkpoint/best_model.pth \
#   --text-ckpt /root/checkpoint/RS5M_ViT-B-32.pt \
#   --out seg_pred_georsclip_vitb32_selfdistill_finetuned_no_featup.png

python demo_georsclip_l_no_featup.py \
  --img demo/oem_koeln_50.tif \
  --name-file ./configs/cls_openearthmap.txt \
  --device cuda \
  --out seg_pred_georsclip_vitb32_segearthov_original_no_featup.png
  # --out seg_pred_georsclip_vitb32_segearthov_finetuned_no_featup.png