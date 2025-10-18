# LibraKAN Framework

This repository implements the **LibraKAN** architecture and its baselines (KAN, KAF, MLP) across multiple levels of evaluation:

- 🧠 `implicit_nf/` — Implicit Neural Fields (coordinate reconstruction on DIV2K, Kodak24, Urban100)
- ⚖️ `kanbefair/` — Benchmark suite for fair KAN-family comparison (used in small-scale experiments)
- 🔬 `libra_ablation/` — Full ablation studies on LibraKAN design (MNIST/FMIST classification & implicit regression)
- 🌍 `libra_downstream/` — Large-scale dense prediction tasks (COCO detection, ADE20K segmentation)
- 📄 `README.md` — Project overview & usage guide (this file)

---

## 🔧 Environment Setup

Recommended Python ≥ 3.10 (tested on CUDA 11.8, PyTorch ≥ 2.3)

```bash
conda create -n librakan python=3.10 -y
conda activate librakan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm scikit-image einops pillow tabulate matplotlib pandas
```

---

## 📁 Folder Overview

| Folder | Purpose | Notes |
|:-------|:---------|:------|
| `implicit_nf/` | Implicit neural field reconstruction (image super-res / signal fitting) | contains spectral design & NUFFT-based LibraKAN core |
| `kanbefair/` | Baseline comparison for MLP, KAN, KAF, FAN | shared dataset loaders and configs |
| `libra_ablation/` | Core ablation experiments (MNIST, FMNIST) | supports ρ, λ, p-sweep and ES kernel toggles |
| `libra_downstream/` | Dense prediction (COCO detection, ADE20K segmentation) | integrated into Faster/Mask R-CNN and UPerNet heads |
| `README.md` | this documentation | generated version for GitHub |

---

## 🚀 Quick Start

### 1️⃣ LibraKAN Ablation (MNIST)

```bash
cd libra_ablation

python ablation.py --task cls --dataset mnist --mixer librakan   --width 256 --depth 2 --epochs 10   --lr 1e-3 --rho 0.2 --tau 1e-3 --p_sparse 0.6   --spectral_scale 0.6 --es_beta 8.0 --no_cos   --alpha_sparse 1e-5 --alpha_balance 1e-4 --grad_clip 0.7   --out runs/cls_libra
```

**Baselines:**
```bash
python ablation.py --mixer kaf --dataset mnist --out runs/kaf
python ablation.py --mixer mlp --dataset mnist --out runs/mlp
```

### 2️⃣ Implicit Neural Field Reconstruction

```bash
cd implicit_nf

python train_nf.py --image data/kodim07.png --mixer librakan   --width 256 --depth 8 --epochs 20 --batch 4096   --lr 1e-3 --rho 0.2 --tau 1e-3 --p_sparse 0.6   --spectral_scale 0.6 --es_beta 7.0 --out runs/kodim07
```

### 3️⃣ Large-scale Downstream Evaluation (Optional)

```bash
cd libra_downstream
python train_coco.py     # Faster/Mask R-CNN + LibraKAN
python train_ade20k.py   # UPerNet + LibraKAN
```

---

## ⚙️ LibraKAN Key Parameters

| Arg | Meaning | Typical |
|:----|:--------|:--------|
| `--rho` | spectral sparsity ratio | 0.2–0.4 |
| `--tau` | base soft-threshold λ | 1e-3 |
| `--p_sparse` | p-norm exponent (Lp shrinkage) | 0.4–0.6 |
| `--spectral_scale` | frequency magnitude scaling | 0.5–0.8 |
| `--es_beta` | ES window sharpness | 6–8 |
| `--no_cos` | disable cosine pair in spectral basis | yes (for stability) |
| `--alpha_sparse` | sparsity penalty weight | 1e-5–5e-5 |
| `--alpha_balance` | local/spectral branch balance | 1e-4 |

---

## 🧩 Ablation Protocol

We study:
1. Removing soft-thresholding (`--wo_soft`)
2. Uniform vs non-uniform frequencies (`--uniform`)
3. Removing ES kernel (`--wo_es`)
4. Varying spectral sparsity ρ (0.1–0.6)
5. Varying λ (×0.5, ×1.0, ×2.0)
6. Varying Lp norm (p = 0–1)
7. Changing local activations (ReLU / SiLU / GELU)

Metrics: **Top-1**, **PSNR**, **Active-Freq**, **FLOPs**  
Default dataset: MNIST or FMNIST

Example sweep:
```bash
for p in 0 0.2 0.4 0.6 0.8; do
  python ablation.py --mixer librakan --p_sparse $p --out runs/p_$p
done
```

---

## 📊 Example Results

| Variant | Top-1 (%) ↑ | PSNR (dB) ↑ | Active-Freq ↓ | FLOPs (G) ↓ |
|:--|:--|:--|:--|:--|
| LibraKAN (default) | **94.2±0.1** | **35.4±0.2** | 24±3 | 0.48 |
| w/o soft-threshold | 90.8 | 30.6 | 88 | 0.53 |
| uniform freq only | 91.9 | 31.2 | 52 | 0.50 |
| w/o ES kernel | 92.3 | 32.0 | 39 | 0.49 |
| ρ=0.2 | 94.2 | 35.4 | 24 | 0.48 |
| p=0.6 | 94.2 | 35.4 | 24 | 0.48 |
| ReLU / SiLU | 93.3 / 94.0 | 34.4 / 35.1 | 25 / 24 | 0.48 |

---

## 📚 Citation

If you use this framework, please cite:

```bibtex
@article{your2025librakan,
  title={LibraKAN: xxx},
  author={xx, xx, xx},
  journal={XXXXX},
  year={2025}
}
```

---

## 🧠 Notes

- All models are pretrained on ImageNet-1K when transferred to COCO/ADE20K.
- MNIST/FMIST ablations do not require pretraining.
- Active-Freq counts post-threshold spectral coefficients (`|a_k| > 1e-3`).
- FLOPs are per single forward pass at canonical resolution.

---

## 🪪 License

This repository is released under the **MIT License**.
