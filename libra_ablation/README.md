# LibraKAN Ablation (Revised, Stable)

Fixes & features:
1) Removed frequency chunking.
2) Correct Active-Freq for KAF/LibraKAN.
3) Stable LibraKAN (no NaNs): safe init, clamped scales, ES window, eps-normalization, nan_to_num, grad clip.
4) Full ablations: w/o soft, uniform vs non-uniform, w/o ES, varying rho/lambda/p.
