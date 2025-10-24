import argparse, os, math, numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_images
from mixers import make_mixer
from train_nf import ImplicitHead
from metrics import psnr, ssim

def img_to_coords(img):
    H, W, C = img.shape
    yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (HW,2)
    target = torch.tensor(img/255.0).view(-1, C).float()  # (HW,C)
    return coords, target, H, W

def train_on_image(img, mixer_name, libra_kwargs, device, epochs, width, lr):
    coords, target, H, W = img_to_coords(img)
    coords, target = coords.to(device), target.to(device)
    model = ImplicitHead(2, width, 3, mixer_name, make_mixer, libra_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(1, epochs+1):
        opt.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        pred = model(coords).clamp(0,1)
        P = (pred.view(H, W, 3).cpu().numpy()*255.0).round().astype(np.uint8)

    spectrum = None
    active = None
    if mixer_name.lower() == "librakan":
        libra = model.mixer.libra if hasattr(model.mixer, "libra") else None
        if libra is not None and libra.last_abs_z is not None:
            mags = libra.last_abs_z.mean(dim=0).detach().cpu().numpy()
            spectrum = np.sort(mags)[::-1]
            active = int((mags > 1e-3).sum())
    return P, losses, spectrum, active

def plot_figure3(img, recs, out_prefix):
    os.makedirs("results", exist_ok=True)
    # 1) Curves
    plt.figure()
    for k, r in recs.items():
        plt.plot(r['loss'], label=k)
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/curve_{out_prefix}.png")

    # 2) Rank–magnitude for LibraKAN
    if recs.get('LibraKAN', {}).get('spectrum') is not None:
        plt.figure()
        plt.plot(recs['LibraKAN']['spectrum'], label='LibraKAN')
        plt.xlabel("rank"); plt.ylabel("|z| after shrink")
        plt.tight_layout()
        plt.savefig(f"results/spectrum_{out_prefix}.png")

    # 3) Visual panel + PSNR/SSIM
    plt.figure(figsize=(4*(len(recs)+1),4))
    items = [("GT", img)] + list(recs.items())
    for i,(k,v) in enumerate(items):
        plt.subplot(1, len(items), i+1)
        if k=="GT":
            plt.imshow(img); title="GT"
        else:
            plt.imshow(v['img'])
            p = psnr(img, v['img']); s = ssim(img, v['img'])
            title = f"{k}\nPSNR {p:.2f} / SSIM {s:.3f}"
            if 'active' in v and v['active'] is not None:
                title += f"\nAF {v['active']}"
        plt.title(title); plt.axis('off')
    plt.tight_layout(); plt.savefig(f"results/fig3_{out_prefix}.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='kodak24', choices=['kodak24','div2k','urban100'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    # Libra knobs
    ap.add_argument('--F', type=int, default=512)
    ap.add_argument('--spectral_scale', type=float, default=1.0)
    ap.add_argument('--es_beta', type=float, default=6.0)
    ap.add_argument('--es_fmax', type=float, default=None)
    ap.add_argument('--nufft_dim', type=int, default=2)
    ap.add_argument('--lambda_init', type=float, default=0.02)
    ap.add_argument('--lambda_trainable', action='store_true')
    ap.add_argument('--p_trainable', action='store_true')
    ap.add_argument('--p_fixed', type=float, default=0.5)
    ap.add_argument('--models', type=str, default='mlp,kaf,librakan')
    args = ap.parse_args()

    libra_kwargs = dict(
        F=args.F, spectral_scale=args.spectral_scale, es_beta=args.es_beta, es_fmax=args.es_fmax,
        nufft_dim=args.nufft_dim, lambda_init=args.lambda_init, lambda_trainable=args.lambda_trainable,
        p_fixed=args.p_fixed, p_trainable=args.p_trainable, base_activation='gelu',
        use_layernorm=False, dropout=0.0,
    )

    device = torch.device(args.device)
    imgs = load_images(args.dataset)
    # To match Fig.3，我们取前1张或前N张并各自产生 panel
    for idx, img in enumerate(imgs[:3] if args.dataset!='kodak24' else imgs):  # Kodak 全部；其他取前3张示例
        recon = {}
        for name in [m.strip() for m in args.models.split(",")]:
            P, losses, spectrum, active = train_on_image(img, name, libra_kwargs, device, args.epochs, args.width, args.lr)
            recon[name.upper()] = {'img': P, 'loss': losses, 'spectrum': spectrum, 'active': active}
        tag = f"{args.dataset}_{idx+1:02d}"
        plot_figure3(img, recon, out_prefix=tag)

if __name__ == '__main__':
    main()
