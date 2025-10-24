
# LibraKAN Function Approximation Package

Reproduces the *Function Approximation Results* figure by training
MLP/KAN/KAF/LibraKAN on sin and cos.

## Usage
```bash
pip install -r requirements.txt

# CPU
python function_approx_results.py --device cpu --epochs 1000 --seed 41 --save fig_func_approx_extrap.png

# CUDA (if available)
python function_approx_results.py --device cuda --amp --epochs 1000 --seed 41 --save fig_func_approx_extrap.png
```
