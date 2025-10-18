# train.py
import argparse, json, os, random, time
from pathlib import Path
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ====== 你的模型封装（按需改为你的实际路径）======
from models.MLP import MLPClassifierExt
from models.KAF import KAFClassifierExt
from models.KAN import build_kan_classifier
from models.GPKAN import GPKANClassifierExt
from models.FAN import FANClassifierExt
from models.LibraKAN import LibraKANClassifierExt


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Datasets & Transforms
# -----------------------------
MEAN_STD_1CH = (0.5,), (0.5,)
MEAN_STD_3CH = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def build_transforms(image_size: Tuple[int, int], in_ch: int, train: bool, dataset_name: str):
    mean, std = (MEAN_STD_1CH if in_ch == 1 else MEAN_STD_3CH)
    aug = []
    if train and dataset_name.lower() in ["cifar10", "cifar100", "svhn"]:
        # 轻量级增广，与常见设置一致
        aug += [transforms.RandomCrop(image_size, padding=4)]
        if dataset_name.lower() != "svhn":
            aug += [transforms.RandomHorizontalFlip()]
    return transforms.Compose([
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def build_dataset(name: str, data_root: str = "./data"):
    name_l = name.lower()
    if name_l == "mnist":
        train = datasets.MNIST(data_root, train=True, download=True,
                               transform=build_transforms((28, 28), 1, True, name_l))
        test = datasets.MNIST(data_root, train=False, download=True,
                              transform=build_transforms((28, 28), 1, False, name_l))
        input_shape, num_classes = (1, 28, 28), 10

    elif name_l in ["emnist", "emnist-balanced", "emnist_balanced"]:
        train = datasets.EMNIST(data_root, split="balanced", train=True, download=True,
                                transform=build_transforms((28, 28), 1, True, "emnist"))
        test = datasets.EMNIST(data_root, split="balanced", train=False, download=True,
                               transform=build_transforms((28, 28), 1, False, "emnist"))
        input_shape, num_classes = (1, 28, 28), 47

    elif name_l in ["fashionmnist", "fashion-mnist", "fmnist"]:
        train = datasets.FashionMNIST(data_root, train=True, download=True,
                                      transform=build_transforms((28, 28), 1, True, "fashionmnist"))
        test = datasets.FashionMNIST(data_root, train=False, download=True,
                                     transform=build_transforms((28, 28), 1, False, "fashionmnist"))
        input_shape, num_classes = (1, 28, 28), 10

    elif name_l == "kmnist":
        train = datasets.KMNIST(data_root, train=True, download=True,
                                transform=build_transforms((28, 28), 1, True, name_l))
        test = datasets.KMNIST(data_root, train=False, download=True,
                               transform=build_transforms((28, 28), 1, False, name_l))
        input_shape, num_classes = (1, 28, 28), 10

    elif name_l == "svhn":
        train = datasets.SVHN(data_root, split="train", download=True,
                              transform=build_transforms((32, 32), 3, True, name_l))
        test = datasets.SVHN(data_root, split="test", download=True,
                             transform=build_transforms((32, 32), 3, False, name_l))
        input_shape, num_classes = (3, 32, 32), 10

    elif name_l == "cifar10":
        train = datasets.CIFAR10(data_root, train=True, download=True,
                                 transform=build_transforms((32, 32), 3, True, name_l))
        test = datasets.CIFAR10(data_root, train=False, download=True,
                                transform=build_transforms((32, 32), 3, False, name_l))
        input_shape, num_classes = (3, 32, 32), 10

    elif name_l == "cifar100":
        train = datasets.CIFAR100(data_root, train=True, download=True,
                                  transform=build_transforms((32, 32), 3, True, name_l))
        test = datasets.CIFAR100(data_root, train=False, download=True,
                                 transform=build_transforms((32, 32), 3, False, name_l))
        input_shape, num_classes = (3, 32, 32), 100

    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test, input_shape, num_classes


# -----------------------------
# Model Factory
# -----------------------------

def build_model(name: str, input_shape, num_classes, args) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPClassifierExt(
            input_shape=input_shape, num_classes=num_classes,
            hidden_dims=args.hidden, activation=args.activation,
            use_layernorm=args.use_layernorm, dropout=args.dropout
        )
    elif name == "kaf":
        return KAFClassifierExt(
            input_shape=input_shape, num_classes=num_classes,
            hidden_dims=args.hidden,
            num_grids=args.num_grids, activation_expectation=args.act_expect,
            base_activation=(F.gelu if args.base_activation == "gelu" else F.silu),
            use_layernorm=args.use_layernorm, dropout=args.dropout
        )
    elif name.lower() == "kan_py":
        return build_kan_classifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden=args.hidden,
            depth=getattr(args, "depth", 1),
            grid=getattr(args, "num_grids", 5),
            degree=getattr(args, "degree", 3),
            x_range=getattr(args, "x_range", 1.0),
            use_layernorm=getattr(args, "use_layernorm", False),
            p_drop=getattr(args, "dropout", 0.0),
        )
    elif name == "gpkan":
        return GPKANClassifierExt(
            input_shape=input_shape, num_classes=num_classes,
            hidden_dims=args.hidden, mlp_ratio=args.mlp_ratio,
            drop=args.dropout, act_init=args.gpkan_act, use_norm=False
        )
    elif name == "fan":
        return FANClassifierExt(
            input_shape=input_shape, num_classes=num_classes,
            hidden_dims=args.hidden, p_ratio=args.p_ratio,
            base_activation=args.base_activation,
            use_layernorm=args.use_layernorm, dropout=args.dropout,
            w_init_scale=args.w_init_scale
        )
    elif name in ["librakan", "libranet"]:
        return LibraKANClassifierExt(
            input_shape=input_shape, num_classes=num_classes,
            hidden_dims=args.hidden, F=args.F, spectral_scale=args.spectral_scale,
            es_beta=args.es_beta, es_fmax=(None if args.es_fmax <= 0 else args.es_fmax),
            lambda_init=args.lambda_init, lambda_trainable=not args.lambda_fixed,
            l1_alpha=args.l1_alpha, base_activation=args.base_activation,
            use_layernorm=args.use_layernorm, dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# -----------------------------
# Train / Eval
# -----------------------------
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, scaler, device, args) -> Dict[str, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            # LibraKAN 额外 L1 正则
            if hasattr(model, "extra_losses"):
                loss = loss + model.extra_losses().get("l1_spectral", 0.0)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return {"loss": total_loss / n, "acc": total_acc / n}


@torch.no_grad()
def evaluate(model, loader, device, args) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return {"loss": total_loss / n, "acc": total_acc / n}

import csv

def append_result_to_csv(args, best, num_params, out_csv="./results.csv"):
    """
    将一次实验的 summary 附加保存到 csv。
    Columns: dataset, model, hidden, num_params, top1_acc
    """
    fieldnames = ["dataset", "model", "hidden", "num_params", "top1_acc"]

    row = {
        "dataset": args.dataset,
        "model": args.model,
        "hidden": args.hidden,
        "num_params": num_params,
        "top1_acc": round(best["acc"], 4),
    }

    # 保证多进程安全 append
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"✅ Appended result to {out_csv}: {row}")

# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    # basic
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--use-layernorm", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dataset", type=str, required=True,
                   choices=["mnist", "emnist", "fashionmnist", "fmnist", "kmnist", "svhn", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, required=True,
                   choices=["mlp", "kaf", "kan_py", "kan", "gpkan", "fan", "librakan", "libranet"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out", type=str, default="./runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=-1)  # -1 -> 根据模型默认
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[256])
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu", "silu"])
    p.add_argument("--base-activation", type=str, default="gelu", choices=["gelu", "silu", "relu"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)

    # KAF
    p.add_argument("--num-grids", type=int, default=16)
    p.add_argument("--act-expect", type=float, default=1.64)

    # KAN
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--x-range", type=float, default=1.0)

    # GPKAN
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--gpkan-act", type=str, default="gelu", choices=["gelu", "swish"])

    # FAN
    p.add_argument("--p-ratio", type=float, default=0.25)
    p.add_argument("--w-init-scale", type=float, default=1.0)

    # LibraKAN
    p.add_argument("--F", type=int, default=128)
    p.add_argument("--spectral-scale", type=float, default=1.0)
    p.add_argument("--es-beta", type=float, default=8.0)
    p.add_argument("--es-fmax", type=float, default=-1.0)  # <=0 表示自动
    p.add_argument("--lambda-init", type=float, default=0.01)
    p.add_argument("--lambda-fixed", action="store_true")
    p.add_argument("--l1-alpha", type=float, default=5e-4)

    args = p.parse_args()

    # epochs 默认策略：KAN=100，其他=40
    if args.epochs <= 0:
        args.epochs = 100 if args.model in ["kan", "kan_py"] else 40

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    # data
    train_set, test_set, input_shape, num_classes = build_dataset(args.dataset, args.data_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model = build_model(args.model, input_shape, num_classes, args).to(device)

    # parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.model}|{args.dataset}] trainable params: {num_params} ({num_params / 1e6:.3f} M)")

    # opt & sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # 余弦退火（简单稳）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # train
    best = {"acc": 0.0, "epoch": -1}
    log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        test_stats = evaluate(model, test_loader, device, args)
        scheduler.step()

        if test_stats["acc"] > best["acc"]:
            best = {"acc": test_stats["acc"], "epoch": epoch}

        row = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "train_acc": round(train_stats["acc"], 6),
            "test_loss": round(test_stats["loss"], 6),
            "test_acc": round(test_stats["acc"], 6),
            "lr": optimizer.param_groups[0]["lr"]
        }
        # LibraKAN 统计
        if hasattr(model, "stats"):
            try:
                row["librakan_stats"] = model.stats()
            except Exception:
                pass
        log.append(row)
        print(f"[{args.model}|{args.dataset}] epoch {epoch:03d}  "
              f"train_acc={row['train_acc']:.4f}  test_acc={row['test_acc']:.4f}  "
              f"best={best['acc']:.4f}@{best['epoch']}")
        append_result_to_csv(args, best, num_params, out_csv=os.path.join(args.out, "results.csv"))

    dt = time.time() - t0
    # save metrics
    run_id = f"{args.model}_{args.dataset}_seed{args.seed}"
    out_file = Path(args.out) / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump({
            "args": vars(args),
            "best": best,
            "log": log,
            "elapsed_sec": dt,
            "num_params": sum(p.numel() for p in model.parameters())
        }, f, indent=2)
    print(f"✅ Done. Best test_acc={best['acc']:.4f} at epoch {best['epoch']}. "
          f"Saved to {out_file}")


if __name__ == "__main__":
    main()
