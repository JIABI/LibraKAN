pip install -e .

cd experiments/4.2_kanbefair    
python train.py --dataset cifar10 --mixer mlp --epochs 40


python train.py --dataset cifar10 --mixer librakan --nufft_dim 1 --p_trainable --lambda_trainable --F 256


python train.py --dataset cifar100 --mixer librakan --nufft_dim 2 --p_fixed 0.6 --F 256

python train.py --dataset mnist --mixer kan
python train.py --dataset cifar10 --mixer kaf
python train.py --dataset svhn  --mixer kat
