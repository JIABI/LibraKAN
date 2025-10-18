#for classification top1
python ablation.py --dataset mnist --task cls --mixer librakan \
  --epochs 10 --width 128 --depth 2 --lr 1e-3 --rho 0.15 --tau 2e-3 --p_sparse 1.0 --spectral_scale 0.5 --es_beta 8.0 --no_cos\
  --alpha_sparse 1e-5 --alpha_balance 1e-4 --out runs/cls_libra

# for PSNR
python ablation.py --dataset fmnist --task imp --mixer librakan \
  --epochs 20 --width 128 --depth 4 --rho 0.3 --tau 1e-3 --p_sparse 0.6 \
  --out runs/imp_libra

# for ablation study for librakan wo soft
python ablation.py --dataset mnist --task cls --mixer librakan --wo_soft\
  --epochs 10 --width 128 --depth 2 --rho 0.3 --tau 1e-3 --p_sparse 1.0 \
  --out runs/wo_soft_libra

# for uniform
python ablation.py --dataset mnist --task cls --mixer librakan --uniform\
  --epochs 10 --width 128 --depth 2 --rho 0.3 --tau 1e-3 --p_sparse 1.0 \
  --out runs/uniform_libra

# for es
python ablation.py --dataset mnist --task cls --mixer librakan --wo_es\
  --epochs 10 --width 128 --depth 2 --rho 0.3 --tau 1e-3 --p_sparse 1.0 \
  --out runs/wo_es_libra

# for p value
for r in 0 0.2 0.4 0.6 0.8 1; do
  python ablation.py --dataset mnist --task cls --mixer librakan --rho $r \
  --epochs 10 --width 128 --depth 2 --tau 1e-3 --p_sparse 1.0 \
  --out runs/sweep_rho/r$r
done

# for lambda
for tau in 5e-4 8e-4 1e-3 2e-3; do
  python ablation.py --dataset mnist --task cls --mixer librakan --tau $tau \
  --epochs 10 --width 128 --depth 2 --rho 1 --p_sparse 1.0 \
  --out runs/sweep_rho/r$r
done