#!/usr/bin/env bash

set -e

DATA=./data

LOG=./logs

# 2) KAF
#python train_cifar10_resnet18.py --data-root $DATA --mixer kaf --mixer-module models.KAF --mixer-class KAFMixerExt  --batch-size 256 --hidden-dim 512 \
#  --kaf-activation-expectation 1.64 --lr 1e-3 --weight-decay 5e-4 --log-dir $LOG \
#  --save --model-out ckpt_kaf.pth --epochs 100



# 1) MLP
# python train_cifar10_resnet18.py --data-root $DATA --mixer mlp --epochs 100 --batch-size 256 --hidden-dim 512 --lr 1e-3 --weight-decay 5e-4 --log-dir $LOG --save --model-out ckpt_mlp.pth

# 6) LibraKAN（可选稀疏率 rho）
python train_cifar10_resnet18.py --mixer librakan --mixer-module models.LibraKAN --mixer-class LibraKANMixerExt --l1-sparsity-weight 1e-5  --batch-size 256 --hidden-dim 512 --lr 1e-3 --weight-decay 5e-4 --epochs 100 --save --model-out ckpt_kan.pth


# 5) KAN (按 KanBench 习惯延长到 100 epochs)

python train_cifar10_resnet18.py --data-root $DATA --mixer kan --kan-epochs 100 --batch-size 128 --hidden-dim 512 --lr 1e-3 --weight-decay 5e-4 --log-dir $LOG --save --model-out ckpt_kan.pth

