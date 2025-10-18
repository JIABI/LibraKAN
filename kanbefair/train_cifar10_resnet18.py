#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, time, math, random, importlib, csv
from pathlib import Path
import importlib
import importlib.util

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / B)).item())
        return res


import inspect
import importlib, importlib.util
from pathlib import Path
import sys

class KAFAdapter(nn.Module):
    """
    适配多种 KAF 写法为统一接口：(N, dim)->(N, dim)。
    支持构造函数:
      - KAF(dim)
      - KAF(in_dim=dim, out_dim=dim)
      - KAF(input_dim=dim, output_dim=dim)
      - KAF(d_model=dim)  # 会回退成 KAF(dim)
    """

    def __init__(self, module_path: str, class_name: str, dim: int, **kwargs):
        super().__init__()
        mod = self._load_module(module_path)
        if not hasattr(mod, class_name):
            raise ImportError(f"Class '{class_name}' not found in module '{module_path}'")
        KAFCls = getattr(mod, class_name)

        # 读取签名，自动映射常见参数名
        try:
            sig = inspect.signature(KAFCls.__init__)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()

        # 常见名字尝试
        ctor_kwargs = dict(**kwargs)
        used = False
        # 1) in_dim/out_dim 风格
        if {"in_dim", "out_dim"} <= params:
            ctor_kwargs.update(in_dim=dim, out_dim=dim)
            self.layer = KAFCls(**ctor_kwargs);
            used = True
        # 2) input_dim/output_dim 风格
        elif {"input_dim", "output_dim"} <= params:
            ctor_kwargs.update(input_dim=dim, output_dim=dim)
            self.layer = KAFCls(**ctor_kwargs);
            used = True
        # 3) d_model 风格（如 transformer 类）
        elif "d_model" in params:
            ctor_kwargs.update(d_model=dim)
            self.layer = KAFCls(**ctor_kwargs);
            used = True
        # 4) 仅一个 dim
        else:
            try:
                self.layer = KAFCls(dim, **ctor_kwargs);
                used = True
            except TypeError:
                # 最后尝试无参
                self.layer = KAFCls(**ctor_kwargs);
                used = True

        if not used:
            raise TypeError(f"Cannot instantiate {class_name} with auto-mapped args; "
                            f"please adjust KAFAdapter mapping.")

    def _load_module(self, module_path: str):
        """
        支持两种传法：
          - 包路径: 'models.KAF' / 'KAF'
          - 绝对/相对文件路径: '/abs/.../KAF.py'
        """
        if module_path.endswith(".py"):
            p = Path(module_path).expanduser().resolve()
            spec = importlib.util.spec_from_file_location("kaf_user_mod", str(p))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load KAF from file: {p}")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
        else:
            # 确保项目根与 models 在 sys.path
            root = Path(__file__).parent.resolve()
            if str(root) not in sys.path: sys.path.insert(0, str(root))
            if str(root / "models") not in sys.path: sys.path.insert(0, str(root / "models"))
            return importlib.import_module(module_path)

    def forward(self, x):
        return self.layer(x)

# 放在 train_cifar10_resnet18.py 顶部 imports 之后

import inspect

import torch.nn as nn


class KANAdapter(nn.Module):
    """

    Wrap pykan.kan.MultKAN into a (N, dim)->(N, dim) mixer layer, auto-mapping ctor kwargs.

    width is fixed to [dim, dim].

    """

    def __init__(self, dim: int, grid: int = 10, degree: int = 3, span: float = 2.0,

                 base_activation: str = "gelu"):

        super().__init__()

        try:

            from pykan.kan.MultKAN import MultKAN

        except Exception as e:

            raise ImportError(

                "Cannot import pykan.kan.MultKAN. Ensure pykan is installed or on PYTHONPATH."

            ) from e

        width = [dim, dim]

        # --- Build a kwargs dict and adapt to the actual signature dynamically ---

        sig = inspect.signature(MultKAN.__init__)

        valid = set(sig.parameters.keys())

        # candidate names for each semantic arg

        argmap = {

            "grid": ["grid", "G", "num_grid", "n_grid", "grids"],

            "degree": ["k", "degree", "order"],

            "span": ["span", "range", "x_range", "spline_range"],

            "base_activation": ["base_activation", "activation", "act"],

            "width": ["width", "layers", "channels"],

        }

        # helper: pick the first available name for a semantic key

        def pick_name(key_list):

            for n in key_list:

                if n in valid:
                    return n

            return None

        ctor_kwargs = {}

        # width

        name = pick_name(argmap["width"])

        if name:

            ctor_kwargs[name] = width

        else:

            # 无论如何 width 必须传；若构造器没有该名，直接按位置参数 fallback

            pass

        # grid

        name = pick_name(argmap["grid"])

        if name: ctor_kwargs[name] = grid

        # degree/k

        name = pick_name(argmap["degree"])

        if name: ctor_kwargs[name] = degree

        # span/range

        name = pick_name(argmap["span"])

        if name: ctor_kwargs[name] = span

        # base activation

        name = pick_name(argmap["base_activation"])

        if name: ctor_kwargs[name] = base_activation

        # Try two strategies: keyword call (preferred), else positional fallback (width only)

        try:

            self.net = MultKAN(**ctor_kwargs)

        except TypeError:

            # positional fallback: assume signature like MultKAN(width, grid=..., k=..., range=..., ...)

            self.net = MultKAN(width, **{k: v for k, v in ctor_kwargs.items() if k != pick_name(argmap["width"])})

    def forward(self, x):

        return self.net(x)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)



# ----------------------------
# Pluggable Mixer Loader
# ----------------------------
# 放在文件顶部附近

CANDIDATE_CLASSNAMES = {
    "mlp": ["MLP", "Mlp", "MlpBlock"],
    "kaf": ["KAF", "Kaf"],
    "gpkan": ["GPKAN", "GpKAN", "GP_KAN"],
    "fan": ["FAN", "Fan"],
    "kan": ["KAN", "Kan"],
    "librakan": ["LibraKAN", "LibraKan", "Libra_KAN"],
}


def _import_module_from_path(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


class BuiltinMLPMixer(nn.Module):
    """(N, dim)->(N, dim) 两层 MLP（GELU），作为 MLP mixer 兜底实现。"""

    def __init__(self, dim: int, hidden: int = None, act: str = "gelu", dropout: float = 0.0):
        super().__init__()

        hidden = hidden or dim

        act_layer = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(act.lower(), nn.GELU)

        self.net = nn.Sequential(

            nn.Linear(dim, hidden, bias=True),

            act_layer(),

            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),

            nn.Linear(hidden, dim, bias=True),

        )

    def forward(self, x): return self.net(x)


def load_mixer(name: str, dim: int, **kwargs) -> nn.Module:
    # ---- 显式覆盖优先：--mixer-module 与 --mixer-class ----
    mixer_module = kwargs.pop("mixer_module", None)  # e.g., "models.librakan" 或 "/abs/path/to/librakan.py"
    mixer_class = kwargs.pop("mixer_class", None)  # e.g., "LibraKAN"
    if mixer_module and mixer_class:
        # 1) 直接给了 .py 文件路径
        import importlib
        if mixer_module.endswith(".py"):
            from pathlib import Path
            path = Path(mixer_module).expanduser().resolve()
            spec = importlib.util.spec_from_file_location("user_mixer", str(path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load mixer module from file: {path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            # 2) 给的是包路径（如 models.librakan）
            #    确保项目根/ models 在 sys.path 里，否则 import_module 找不到
            import sys
            from pathlib import Path
            ROOT = Path(__file__).parent.resolve()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            if str(ROOT / "models") not in sys.path:
                sys.path.insert(0, str(ROOT / "models"))
            mod = importlib.import_module(mixer_module)

        if not hasattr(mod, mixer_class):
            raise ImportError(f"Class '{mixer_class}' not found in module '{mixer_module}'")
        cls = getattr(mod, mixer_class)
        try:
            return cls(dim, **kwargs)
        except TypeError:
            return cls(dim)
    key = name.lower()
    if key == "mlp":
        # 先尝试用户模块
        try:
            modnames = ["MLP", "mlp", "models.MLP", "models.mlp"]
            for mod in modnames:
                try:
                    import importlib
                    module = importlib.import_module(mod)
                    for cls_name in ["MLP", "Mlp", "MlpBlock"]:
                        if hasattr(module, cls_name):
                            cls = getattr(module, cls_name)
                            try:
                                return cls(dim, **kwargs)
                            except TypeError:
                                return cls(dim)
                except ModuleNotFoundError:
                    pass
        except Exception:
            pass
        # 兜底：内置 MLP
        hidden = int(kwargs.pop("hidden", dim))
        act = kwargs.pop("act", "gelu")
        dropout = float(kwargs.pop("dropout", 0.0))
        return BuiltinMLPMixer(dim=dim, hidden=hidden, act=act, dropout=dropout)

    # --- 专门处理 KAN：用适配器把 MultKAN 包成 dim->dim ---
    if key == "kan":
        # 从 kwargs 里取网格/阶数/范围，给默认值
        grid = int(kwargs.pop("kan_grid", kwargs.pop("grid", 10)))
        degree = int(kwargs.pop("kan_degree", kwargs.pop("degree", 3)))
        span = float(kwargs.pop("kan_span", kwargs.pop("kan_range", 2.0)))
        base_act = str(kwargs.pop("kan_base_activation", "gelu")).lower()
        return KANAdapter(dim=dim, grid=grid, degree=degree, span=span,
                          base_activation=base_act)
    elif key == "kaf":
        # 优先使用 CLI 显式指定（--mixer-module / --mixer-class）
        mixer_module = kwargs.pop("mixer_module", None)  # 例如 'models.KAF' 或 '/abs/path/KAF.py'
        mixer_class = kwargs.pop("mixer_class", None)  # 例如 'KAF'
        if mixer_module and mixer_class:
            return KAFAdapter(module_path=mixer_module, class_name=mixer_class, dim=dim, **kwargs)

        # 否则自动尝试若干常见位置 + 类名
        candidates_mod = ["models.KAF", "models.kaf", "KAF", "kaf"]
        candidates_cls = ["KAF", "Kaf"]

        # 1) 包路径导入
        for modname in candidates_mod:
            try:
                import importlib
                module = importlib.import_module(modname)
                for cls_name in candidates_cls:
                    if hasattr(module, cls_name):
                        # 封一层适配器确保 dim->dim
                        return KAFAdapter(module_path=modname, class_name=cls_name, dim=dim, **kwargs)
            except ModuleNotFoundError:
                pass

        # 2) 本地文件兜底（当前目录与 models/）
        from pathlib import Path
        here = Path(__file__).parent
        for stem in ["KAF", "kaf"]:
            for p in [here / f"{stem}.py", here / "models" / f"{stem}.py"]:
                if p.exists():
                    return KAFAdapter(module_path=str(p), class_name="KAF", dim=dim, **kwargs)

        raise ImportError("Could not import KAF. "
                          "Use --mixer-module / --mixer-class or ensure models/KAF.py defines class KAF.")

    # --- 其余类型按原逻辑动态导入 ---
    keymap = {
        "mlp": ["MLP", "Mlp", "MlpBlock"],
        "kaf": ["KAF", "Kaf"],
        "gpkan": ["GPKAN", "GpKAN", "GP_KAN"],
        "fan": ["FAN", "Fan"],
        "librakan": ["LibraKAN", "LibraKan", "Libra_KAN"],
    }
    if key not in keymap:
        raise ValueError(f"Unknown mixer '{name}'. Options: {list(keymap.keys()) + ['kan']}")

    # 优先尝试同名模块（同目录），再尝试大小写变种
    import importlib, importlib.util
    from pathlib import Path
    candidates_mod = [name, name.lower(), name.upper(), name.capitalize(),
                      ("LibraKAN" if key == "librakan" else name)]
    # 1) import by module name
    for mod in candidates_mod:
        try:
            module = importlib.import_module(mod)
            for cls_name in keymap[key]:
                if hasattr(module, cls_name):
                    cls = getattr(module, cls_name)
                    try:
                        return cls(dim, **kwargs)
                    except TypeError:
                        return cls(dim)
        except ModuleNotFoundError:
            pass

    # 2) import by local file path
    here = Path(__file__).parent
    for stem in candidates_mod:
        p = here / f"{stem}.py"
        if p.exists():
            spec = importlib.util.spec_from_file_location(stem, str(p))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            for cls_name in keymap[key]:
                if hasattr(module, cls_name):
                    cls = getattr(module, cls_name)
                    try:
                        return cls(dim, **kwargs)
                    except TypeError:
                        return cls(dim)

    raise ImportError(f"Could not import mixer '{name}'. Checked modules {candidates_mod} and local files.")


# ----------------------------
# ResNet18 backbone + pluggable head
# ----------------------------
class ResNet18WithMixer(nn.Module):
    """
    Backbone: torchvision ResNet-18 (convs unchanged).
    Head: GlobalAvgPool -> Linear(C, H) -> Mixer(H) -> Linear(H, num_classes)
    Mixer must preserve shape (N, H).
    """
    def __init__(self, num_classes: int, mixer_name: str, hidden_dim: int = 512,
                 mixer_kwargs: dict = None):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        C = self.backbone.fc.in_features  # 512
        # Replace the original fc with identity; we'll do our own head.
        self.backbone.fc = nn.Identity()

        self.fc_in = nn.Linear(C, hidden_dim)
        self.act = nn.Identity()  # the nonlinearity is handled by the mixer
        mixer_kwargs = mixer_kwargs or {}
        self.mixer = load_mixer(mixer_name, hidden_dim, **mixer_kwargs)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)          # (N, 512)
        h = self.fc_in(feats)             # (N, H)
        h = self.mixer(h)                 # (N, H)
        logits = self.fc_out(h)           # (N, num_classes)
        return logits

# ----------------------------
# Data (CIFAR-10 by default)
# ----------------------------
def build_cifar10(data_root, train, img_size=32):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    ds = datasets.CIFAR10(root=data_root, train=train, transform=transform, download=True)
    return ds

# ----------------------------
# Scheduler: cosine with warmup
# ----------------------------
def cosine_lr(optimizer, base_lr, warmup_epochs, total_epochs, iters_per_epoch):
    warmup_iters = warmup_epochs * iters_per_epoch
    total_iters = total_epochs * iters_per_epoch

    def lr_lambda(it):
        if it < warmup_iters:
            return float(it + 1) / float(max(1, warmup_iters))
        progress = (it - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, print_freq=50):
    model.train()
    loss_meter, acc1_meter = AverageMeter(), AverageMeter()
    start = time.time()

    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy(logits, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1, images.size(0))

        if (it + 1) % print_freq == 0:
            elapsed = time.time() - start
            print(f"Epoch [{epoch}] Iter [{it+1}/{len(loader)}] "
                  f"Loss {loss_meter.avg:.4f}  Acc@1 {acc1_meter.avg:.2f}  "
                  f"{images.size(0)*print_freq/elapsed:.1f} img/s")
            start = time.time()
    return loss_meter.avg, acc1_meter.avg

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter, acc1_meter = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(images)
            ce = F.cross_entropy(logits, targets)
            reg = 0.0
            if hasattr(model, "mixer") and hasattr(model.mixer, "extra_losses"):
                for v in model.mixer.extra_losses().values():
                    reg = reg + v
            loss = ce + 0 * reg
        acc1 = accuracy(logits, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1, images.size(0))
    return loss_meter.avg, acc1_meter.avg

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="ResNet18 w/ pluggable feature mixer on CIFAR-10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="hidden width H of the head (after GAP).")
    parser.add_argument("--mixer", type=str, default="mlp",
                        choices=["mlp", "kaf", "gpkan", "fan", "kan", "librakan"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--model-out", type=str, default="./ckpt.pth")
    parser.add_argument("--kan-epochs", type=int, default=100,
                        help="if mixer==kan, override epochs to this (to match KanBench).")
    # pass-through kwargs for specific mixers, e.g., --kaf-activation-expectation 1.64
    parser.add_argument("--kaf-activation-expectation", type=float, default=1.64)
    parser.add_argument("--fan-p-ratio", type=float, default=0.25)
    parser.add_argument("--sparsity-rho", type=float, default=None,
                        help="LibraKAN spectral sparsity (e.g., 0.4). If None, mixer default.")
    parser.add_argument("--mixer-module", type=str, default=None,
                        help="Override module path for mixer, e.g. 'models.librakan' or '/abs/path/to/librakan.py'")
    parser.add_argument("--mixer-class", type=str, default=None,
                        help="Override class name for mixer, e.g. 'LibraKAN'")
    parser.add_argument("--l1-sparsity-weight", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Datasets
    train_set = build_cifar10(args.data_root, train=True)
    test_set  = build_cifar10(args.data_root, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    num_classes = 10

    # Mixer kwargs per type
    mixer_kwargs = {}
    if args.mixer == "kaf":
        mixer_kwargs.update(dict(activation_expectation=args.kaf_activation_expectation))
    elif args.mixer == "fan":
        mixer_kwargs.update(dict(p_ratio=args.fan_p_ratio))
    elif args.mixer == "librakan":
        if args.sparsity_rho is not None:
            mixer_kwargs["rho"] = args.sparsity_rho
    # 关键：把 CLI 的覆盖选项传给 loader
    mixer_kwargs["mixer_module"] = args.mixer_module
    mixer_kwargs["mixer_class"] = args.mixer_class
    model = ResNet18WithMixer(num_classes=num_classes,
                              mixer_name=args.mixer,
                              hidden_dim=args.hidden_dim,
                              mixer_kwargs=mixer_kwargs).to(device)

    # Optim / Sched
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_epochs = args.kan_epochs if args.mixer == "kan" else args.epochs
    scheduler = cosine_lr(optimizer, base_lr=args.lr,
                          warmup_epochs=args.warmup_epochs,
                          total_epochs=total_epochs,
                          iters_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler('cuda')

    # Logs
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    csv_file = os.path.join(args.log_dir, f"log_{args.mixer}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc1", "test_loss", "test_acc1", "lr"])

    best_acc = 0.0
    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        lr_cur = scheduler.get_last_lr()[0]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{test_loss:.4f}", f"{test_acc:.2f}", f"{lr_cur:.6f}"])
        print(f"[Epoch {epoch:03d}/{total_epochs}] "
              f"train_acc {train_acc:.2f}  test_acc {test_acc:.2f}  best {best_acc:.2f}")

        if test_acc > best_acc:
            best_acc = test_acc
            if args.save:
                torch.save({"epoch": epoch,
                            "state_dict": model.state_dict(),
                            "best_acc1": best_acc,
                            "args": vars(args)}, args.model_out)

    print(f"Done. Best Acc@1 = {best_acc:.2f}")

if __name__ == "__main__":
    main()
