pip install -e .

cd experiments/4.4_implicit_nf

# 1) Kodak24：
python train.py --dataset kodak24 \
  --models mlp,kaf,librakan \
  --epochs 200 --width 256 --nufft_dim 2 --p_trainable --lambda_trainable

# 2) DIV2K --epochs、--width）
python train.py --dataset div2k \
  --models mlp,kaf,librakan \
  --nufft_dim 2 --p_fixed 0.5 --F 512

# 3) Urban100
python train.py --dataset urban100 \
  --models mlp,kaf,librakan \
  --nufft_dim 2 --p_trainable --lambda_trainable
