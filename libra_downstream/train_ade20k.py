import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from models.segmentation_mixer import SegModel
from utils.metrics import compute_miou
from utils.complexity import count_params_m, get_flops_g

class ADE20KDataset(Dataset):
    def __init__(self, root, split="training", size=512):
        self.root = root; self.split=split; self.size=size
        img_dir = os.path.join(root, "images", split)
        ann_dir = os.path.join(root, "annotations", split)
        self.items = []
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg",".png")):
                imgp = os.path.join(img_dir, fname)
                annp = os.path.join(ann_dir, fname.replace(".jpg",".png"))
                if os.path.exists(annp):
                    self.items.append((imgp, annp))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        imgp, annp = self.items[idx]
        img = Image.open(imgp).convert("RGB").resize((self.size,self.size), Image.BILINEAR)
        ann = Image.open(annp).convert("L").resize((self.size,self.size), Image.NEAREST)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
        ann = torch.from_numpy(np.array(ann)).long()
        return img, ann

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", type=str, choices=["mlp","kaf","librakan"], default="mlp")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="runs/ade20k_run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_set = ADE20KDataset(args.data_root, split="training")
    val_set   = ADE20KDataset(args.data_root, split="validation")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)

    model = SegModel(mixer_type=args.mixer, pretrained_backbone=True, num_classes=150).cuda()

    params_m = count_params_m(model)
    try:
        example = torch.randn(1,3,512,512).cuda()
        flops_g = get_flops_g(model, example)
    except Exception:
        flops_g = -1.0

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=255)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}")
        for img, ann in pbar:
            img = img.cuda(); ann = ann.cuda()
            logits = model(img)
            loss = crit(logits, ann)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.item()))

        # eval
        model.eval(); miou_sum=0.0; n=0
        with torch.no_grad():
            for img, ann in tqdm(val_loader, desc="Eval", leave=False):
                img = img.cuda(); ann = ann.cuda()
                logits = model(img)
                pred = torch.argmax(logits, dim=1)
                miou_sum += compute_miou(pred, ann, num_classes=150)
                n += 1
        miou = miou_sum / max(1,n)
        if miou > best:
            best = miou
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"task":"ade20k","mixer":args.mixer,"params_M":params_m,"flops_G":flops_g,"mIoU":best}, f, indent=2)

if __name__ == "__main__":
    main()
