# LibraKAN — Upgraded (2025)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900.svg)](#)

---

## 🧠 Overview

**LibraKAN** is a dual-branch Kolmogorov–Arnold network featuring a **NUFFT–ES spectral branch** and a **local MLP branch**.  
It introduces **learnable spectral sparsity** through trainable shrinkage (`λ`) and fractional exponent (`p`), with end-to-end support for both **1D** and **2D NUFFT** operations.

This repository contains **the upgraded 2025 release**, aligned with the latest method implementation and covering **Experiments 4.1–4.6** in the paper.

---

## 🧩 Architecture Diagram

<p align="center">
  <img src="docs/architecture.svg" alt="LibraKAN Architecture" width="720"/>
</p>

> The architecture consists of two cooperative paths:  
> ① **Spectral NUFFT–ES branch** learns non-uniform frequency representations with differentiable exponential-sine kernels;  
> ② **Local MLP branch** preserves fine-grained structure and spatial context;  
> A **shrinkage controller** fuses both, enforcing learnable spectral sparsity.

---

## ⚙️ Installation

```bash
conda create -n librakan python=3.10 -y
conda activate librakan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

Optional third-party baselines:
```bash
pip install git+https://github.com/kolmogorovArnoldFourierNetwork/KAF.git
pip install git+https://github.com/KindXiaoming/pykan.git
pip install git+https://github.com/Adamdad/kat.git
```

---

## 📁 Project Structure

```
librakan_act/
  ├── librakan.py
  ├── nufft_es.py
  ├── shrinkage.py
  └── __init__.py

experiments/
  ├── 4.1_main_results/
  ├── 4.2_kanbefair/
  ├── 4.3_libra_downstream/
  ├── 4.4_implicit_nf/
  ├── 4.5_libra_ablation/
  └── 4.6_Function Approximation/

pyproject.toml
requirements.txt
README.md
```

---

## 🔬 Core Method Components

| Component | Description |
|------------|-------------|
| NUFFT–ES | Differentiable Non-Uniform FFT with Exponential-Sine window |
| Shrinkage | Learnable λ and p (`p = sigmoid(plogit)`) for adaptive sparsity |
| Active-Freq | Counts spectral coefficients |aₖ| > 1e-3 |
| Compatibility | Drop-in mixer for MLP-based backbones |

### Example usage
```python
from librakan_act import make_librakan_mixer
mixer = make_librakan_mixer(in_dim=256, hidden_dim=256,
                            F=512, nufft_dim=2,
                            lambda_init=0.02, lambda_trainable=True,
                            p_trainable=True)
```

---

## 🚀 Experiments

### **4.1 Main Results (Table 1)**

| Model | Dataset | Replacement | Command |
|--------|----------|-------------|----------|
| ResNet-18 | CIFAR-10 | Head MLP | `python resnet18_cifar10.py --mixer librakan` |
| MLP-Mixer/S | ImageNet-1K | Channel MLP | `python mlpmixerS_imagenet.py --mixer librakan` |
| ViT-T/16 | ImageNet-1K | FFN | `python vit_tiny16_imagenet.py --mixer librakan` |
| DeiT-tiny | CIFAR-100 | FFN | `python deit_tiny_cifar100.py --mixer librakan` |

---

### **4.2 KanBeFair (Benchmark Baselines)**
Fair comparison of all MLP-like architectures on CIFAR/MNIST variants.

```bash
cd experiments/4.2_kanbefair
python train.py --dataset cifar10 --mixer librakan --nufft_dim 2 --p_fixed 0.5
```

---

### **4.3 Downstream Vision Tasks**
- COCO Detection (Faster R-CNN)  
- COCO Instance Segmentation (Mask R-CNN)  
- ADE20K Semantic Segmentation (UPerNet-lite)

```bash
cd experiments/4.3_libra_downstream
python fasterrcnn_coco.py --mixer librakan --p_trainable --lambda_trainable
python maskrcnn_coco.py --mixer kaf
python upernet_ade20k.py --mixer librakan --nufft_dim 2
```

---

### **4.4 Implicit Neural Field Reconstruction (Figure 3)**
Reconstruction datasets: **Kodak24**, **DIV2K**, **Urban100**.

Outputs:
- `curve_*.png`: loss vs epoch  
- `spectrum_*.png`: rank–magnitude of |z|  
- `fig3_*.png`: PSNR / SSIM / Active-Freq panel

```bash
cd experiments/4.4_implicit_nf
python train.py --dataset kodak24 --models mlp,kaf,librakan   --epochs 200 --nufft_dim 2 --p_trainable --lambda_trainable
```

---

### **4.5 Ablation Study (Table 6)**
All combinations of `F`, `nufft_dim`, `p`/`λ` learnability, `es_beta` are tested automatically via `run.sh`.

```bash
cd experiments/4.5_libra_ablation
bash run.sh
```

Produces:
- `results/ablation_*.json`  
- `results/table6.csv`  
- `results/ablation_acc_vs_F_*.png`  
- `results/ablation_activefreq_*.png`

---

### **4.6 Function Approximation**
```bash
cd experiments/4.6_Function\ Approximation
python run_funcapprox.py --mixer librakan --nufft_dim 1
```

---

## 📊 Metrics

| Metric | Description |
|:--|:--|
| Top-1 | Classification accuracy |
| PSNR / SSIM | Reconstruction fidelity |
| Active-Freq | Number of active frequencies |
| FLOPs / Params | Model efficiency |

---

## 🧩 Integration Example

```python
from librakan_act import make_librakan_mixer
layer = make_librakan_mixer(in_dim=128, hidden_dim=128, F=256, nufft_dim=1,
                            spectral_scale=1.0, es_beta=6.0,
                            lambda_init=0.02, lambda_trainable=True,
                            p_fixed=0.5, p_trainable=True)
```

---

## 📚 Citation

```bibtex
@article{librakan2025,
  title={CVKAN},
  author={XXX},
  journal={Under Review, XXX 2026},
  year={2025}
}
```

---

## 🪪 License
MIT License © 2025 Libra Research Group.
