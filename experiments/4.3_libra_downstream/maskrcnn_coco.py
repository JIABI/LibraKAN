import argparse, os, json, torch, torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from mixers import make_mixer

class MixerFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, mixer_name, hidden_dim, libra_kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.mixer = make_mixer(mixer_name, hidden_dim, hidden_dim, **libra_kwargs)
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)
    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        x = self.fc1(x).relu_()
        x = self.mixer(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

def get_coco_loaders(root, bs=2):
    train = tv.datasets.CocoDetection(os.path.join(root, "train2017"),
                                      os.path.join(root, "annotations/instances_train2017.json"),
                                      transforms=tv.transforms.ToTensor())
    val = tv.datasets.CocoDetection(os.path.join(root, "val2017"),
                                    os.path.join(root, "annotations/instances_val2017.json"),
                                    transforms=tv.transforms.ToTensor())
    return DataLoader(train, batch_size=bs, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x))),            DataLoader(val, batch_size=bs, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

@torch.no_grad()
def evaluate_proxy(model, loader, device):
    model.eval()
    tot = 0; acc = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out in outputs:
            if len(out.get("masks", []))>0:
                acc += 1
            tot += 1
    return acc/max(tot,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=str, required=True)
    ap.add_argument("--mixer", type=str, default="mlp", choices=["mlp","kaf","kat","fan","kan","librakan"])
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="results")
    # Libra
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--spectral_scale", type=float, default=1.0)
    ap.add_argument("--es_beta", type=float, default=6.0)
    ap.add_argument("--es_fmax", type=float, default=None)
    ap.add_argument("--nufft_dim", type=int, default=1)
    ap.add_argument("--lambda_init", type=float, default=0.01)
    ap.add_argument("--lambda_trainable", action="store_true")
    ap.add_argument("--p_trainable", action="store_true")
    ap.add_argument("--p_fixed", type=float, default=0.5)
    args = ap.parse_args()

    libra_kwargs = dict(F=args.F, spectral_scale=args.spectral_scale, es_beta=args.es_beta, es_fmax=args.es_fmax,
                        nufft_dim=args.nufft_dim, lambda_init=args.lambda_init, lambda_trainable=args.lambda_trainable,
                        p_fixed=args.p_fixed, p_trainable=args.p_trainable, base_activation="gelu",
                        use_layernorm=False, dropout=0.0)

    device = torch.device(args.device)
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MixerFastRCNNPredictor(in_features, 91, args.mixer, args.hidden, libra_kwargs)
    model.to(device)

    train_loader, val_loader = get_coco_loaders(args.coco_root, args.bs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if hasattr(v,'to') else v for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
        proxy = evaluate_proxy(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} proxy_mask={proxy:.4f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"maskrcnn_coco_{args.mixer}.json"), "w") as f:
        json.dump({"task":"coco_instseg_maskrcnn","mixer":args.mixer,"proxy":float(proxy)}, f, indent=2)

if __name__ == "__main__":
    main()
