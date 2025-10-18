# LibraKAN Dense Vision Bench (COCO & ADE20K)

Implements **MLP (baseline)**, **KAF**, and **LibraKAN** mixers as FFN replacements for:
- **COCO 2017**: Faster R-CNN (detection), Mask R-CNN (instance segmentation)
- **ADE20K**: semantic segmentation with a simplified UPerNet-style head

Reports: **Params (M)**, **FLOPs (G)**, **COCO mAP@[0.5:0.95]/AP50**, **ADE20K mIoU**.

## Setup
```bash
conda create -n libradense python=3.10 -y
conda activate libradense
pip install -r requirements.txt
```
Datasets:
- COCO 2017 (`train2017/`, `val2017/`, `annotations/` under $COCO_ROOT)
- ADE20K (`images/{training,validation}`, `annotations/{training,validation}` under $ADE20K_ROOT)

## Run
COCO Detection:
```bash
python train_coco.py --task det --mixer mlp --data_root $COCO_ROOT --out runs/coco_det_mlp
```
COCO Instance Segmentation:
```bash
python train_coco.py --task seg --mixer librakan --data_root $COCO_ROOT --out runs/coco_segm_libra
```
ADE20K Segmentation:
```bash
python train_ade20k.py --mixer kaf --data_root $ADE20K_ROOT --out runs/ade20k_kaf
```

Saved summaries: per-run JSON under output dir; aggregated CSV at `runs/summary.csv`.
