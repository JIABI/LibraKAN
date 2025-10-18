import os, argparse, json
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from models.detection_mixer import build_model
from utils.complexity import count_params_m, get_flops_g


def collate_fn(batch):
    imgs, targets = [], []
    eps = 1e-3  # 防止零宽/零高
    for img_pil, ann in batch:
        # 先转 tensor 拿到 H,W
        img = TF.to_tensor(img_pil)  # (C,H,W), float32 in [0,1]
        _, H, W = img.shape
        imgs.append(img)

        # COCO bbox: [x, y, w, h] -> 转为 [x1, y1, x2, y2]
        boxes, labels = [], []
        for a in ann:
            if "bbox" in a and "category_id" in a:
                x, y, w, h = a["bbox"]
                x1 = float(x)
                y1 = float(y)
                x2 = float(x + w)
                y2 = float(y + h)

                # 裁剪到图像边界
                x1 = max(0.0, min(x1, W - eps))
                y1 = max(0.0, min(y1, H - eps))
                x2 = max(0.0, min(x2, W - eps))
                y2 = max(0.0, min(y2, H - eps))

                # 过滤无效框（必须严格正面积）
                if x2 - x1 > eps and y2 - y1 > eps:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(a["category_id"]))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        targets.append({"boxes": boxes_t, "labels": labels_t})

    return imgs, targets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["det","seg"], default="det")
    ap.add_argument("--mixer", type=str, choices=["mlp","kaf","librakan"], default="mlp")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/coco_run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train = CocoDetection(os.path.join(args.data_root, "train2017"),
                          os.path.join(args.data_root, "annotations/instances_train2017.json"))
    val   = CocoDetection(os.path.join(args.data_root, "val2017"),
                          os.path.join(args.data_root, "annotations/instances_val2017.json"))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_model(task=args.task, mixer=args.mixer, pretrained=True)

    num_classes = 91  # COCO
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if args.task == "seg" and hasattr(model.roi_heads, "mask_predictor"):
        in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden, num_classes)

    model = model.cuda()

    params_m = count_params_m(model)
    try:
        example = [torch.randn(1,3,800,800).cuda()]
        flops_g = get_flops_g(model, example)
    except Exception:
        flops_g = -1.0

    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, targets in pbar:
            imgs = [img.cuda() for img in imgs]
            targets = [{k: v.cuda() for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(v for v in loss_dict.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.item()))
        torch.save(model.state_dict(), os.path.join(args.out, f"epoch_{epoch+1}.pth"))

    # COCO evaluation stub (use official tools for exact mAP in production)
    metrics = {"bbox_mAP": -1.0, "bbox_AP50": -1.0, "segm_mAP": -1.0, "segm_AP50": -1.0}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"task":args.task,"mixer":args.mixer,"params_M":params_m,"flops_G":flops_g, **metrics}, f, indent=2)

if __name__ == "__main__":
    main()
