pip install -e .   

# 1. ResNet-18 on CIFAR-10
cd experiments/4.1_main_results
# baseline MLP
python resnet18_cifar10.py --mixer mlp --epochs 100
# LibraKAN（p/λ ）
python resnet18_cifar10.py --mixer librakan --lambda_trainable --p_trainable --F 256 --nufft_dim 1

# 2. MLP-Mixer/S on ImageNet-1k
# baseline
python mlpmixerS_imagenet.py --imagenet_root /path/to/imagenet --mixer mlp
# LibraKAN（
python mlpmixerS_imagenet.py --imagenet_root /path/to/imagenet --mixer librakan --p_fixed 0.5 --F 256 --nufft_dim 1

# 3. ViT-T/16 on ImageNet-1k
python vit_tiny16_imagenet.py --imagenet_root /path/to/imagenet --mixer mlp
python vit_tiny16_imagenet.py --imagenet_root /path/to/imagenet --mixer librakan --lambda_trainable --p_trainable --F 256

# 4. “MLP KAN(DeiT)” on CIFAR-100
python deit_tiny_cifar100.py --mixer mlp
python deit_tiny_cifar100.py --mixer kan    # 需要 pykan
python deit_tiny_cifar100.py --mixer librakan --p_fixed 0.6 --F 256

# 5 generate table
python aggregate_table1.py




