#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUT=./runs
BS=256
LR=1e-3
SEED=42

WIDTHS=(32 64 128 256 512 1024)
DATASETS=("mnist" "emnist" "fashionmnist" "kmnist" "svhn" "cifar10" "cifar100")
MODELS=("kan_py" "gpkan" "fan" "mlp" "kaf" "librakan")

for DS in "${DATASETS[@]}"; do
  for M in "${MODELS[@]}"; do
    for H in "${WIDTHS[@]}"; do
      echo ">>> Training model=${M} dataset=${DS} hidden=${H}"
      python train_librakan.py --dataset "$DS" --model "$M" --hidden "$H" \
        --batch-size "$BS" --lr "$LR" --out "$OUT" --seed "$SEED" --amp
    done
  done
done