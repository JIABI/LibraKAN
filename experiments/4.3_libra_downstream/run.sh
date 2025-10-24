# 4.3 Downstream (Faster R-CNN / Mask R-CNN / UPerNet-lite)

- Mixers: mlp / kaf / kat / fan / kan / librakan
- LibraKAN knobs shared: --nufft_dim, --p_trainable/--p_fixed, --lambda_trainable/--lambda_init, --F, --spectral_scale, --es_beta, --es_fmax

## COCO Detection (Faster R-CNN)
python fasterrcnn_coco.py --coco_root /path/to/coco --mixer librakan --p_trainable --lambda_trainable

## COCO Instance Segmentation (Mask R-CNN)
python maskrcnn_coco.py --coco_root /path/to/coco --mixer kaf

## ADE20K Semantic Segmentation (UPerNet-lite)
python upernet_ade20k.py --ade20k_root /path/to/ADE20K --mixer librakan --nufft_dim 2 --p_fixed 0.6
