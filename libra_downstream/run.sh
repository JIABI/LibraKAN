# COCO Detection

python train_coco.py --task det --mixer mlp --data_root /home/ubuntu/PycharmProjects/LiberNet/libra_dense_bench/data/coco2017 --out runs/coco_det_mlp

# COCO Instance Segmentation

python train_coco.py --task seg --mixer librakan --data_root /home/ubuntu/PycharmProjects/LiberNet/libra_dense_bench/data/coco2017 --out runs/coco_segm_libra

# ADE20K Semantic Segmentation

python train_ade20k.py --mixer kaf --data_root /home/ubuntu/PycharmProjects/LiberNet/libra_dense_bench/data/ADE20K_2021_17_01 --out runs/ade20k_kaf

