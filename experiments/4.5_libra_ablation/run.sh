#!/usr/bin/env bash
set -e

# Default ablation suite to populate Table 6 quickly.
# Dataset can be overridden with DATASET=...
DATASET=${DATASET:-cifar10}
EPOCHS=${EPOCHS:-40}
BS=${BS:-256}
WIDTH=${WIDTH:-256}
LR=${LR:-3e-4}

# 1) Baselines
python train.py --dataset $DATASET --mixer mlp --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR
python train.py --dataset $DATASET --mixer kaf --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR

# 2) LibraKAN ablations: p fixed values
for P in 0.25 0.5 0.75; do
  python train.py --dataset $DATASET --mixer librakan --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR     --p_fixed $P --F 128 --nufft_dim 1 --lambda_init 0.01
done

# 3) LibraKAN ablations: learnable p and/or lambda
python train.py --dataset $DATASET --mixer librakan --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR   --p_trainable --lambda_init 0.01
python train.py --dataset $DATASET --mixer librakan --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR   --p_trainable --lambda_trainable

# 4) F and dimension
for F in 128 256 512; do
  python train.py --dataset $DATASET --mixer librakan --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR     --p_fixed 0.5 --lambda_init 0.01 --F $F --nufft_dim 1
  python train.py --dataset $DATASET --mixer librakan --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR     --p_fixed 0.5 --lambda_init 0.01 --F $F --nufft_dim 2
done

# 5) ES beta
for BETA in 4.0 6.0 8.0; do
  python train.py --dataset $DATASET --mixer librakan --epochs $EPOCHS --bs $BS --width $WIDTH --lr $LR     --p_fixed 0.5 --lambda_init 0.01 --F 256 --nufft_dim 2 --es_beta $BETA
done

# Aggregate and plots
python aggregate_table6.py
python plot_ablation.py

echo "All done. See results/table6.csv and plots under results/."
