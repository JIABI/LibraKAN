import os, glob, urllib.request, zipfile, tarfile
from PIL import Image
import numpy as np

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _download(url, out_path):
    try:
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception:
        return False

def load_kodak24(root="./data/kodak24"):
    _ensure_dir(root)
    imgs = []
    for i in range(1, 25):
        name = f"kodim{str(i).zfill(2)}.png"
        path = os.path.join(root, name)
        if not os.path.exists(path):
            _download(f"http://r0k.us/graphics/kodak/{name}", path)
        if os.path.exists(path):
            imgs.append(np.array(Image.open(path).convert("RGB")))
    if not imgs:
        raise RuntimeError("Kodak24 not available. Place PNGs under " + root)
    return imgs

def load_div2k(root="./data/DIV2K", subset="valid"):
    # Expect local structure: DIV2K/HR/{0001..0800}.png for train, DIV2K_valid_HR/{0801..0900}.png for valid
    # We avoid auto-download due to size/license. User must place files.
    hr_dirs = {"train":"DIV2K_train_HR","valid":"DIV2K_valid_HR"}
    hr = os.path.join(root, hr_dirs.get(subset,"DIV2K_valid_HR"))
    if not os.path.isdir(hr):
        raise RuntimeError(f"DIV2K HR folder not found: {hr}. Please place images.")
    files = sorted(glob.glob(os.path.join(hr,"*.png")))
    if not files:
        raise RuntimeError("DIV2K found but no PNGs in " + hr)
    return [np.array(Image.open(p).convert("RGB")) for p in files]

def load_urban100(root="./data/Urban100"):
    # Try to download archive if possible; otherwise require manual placement.
    imgs_dir = os.path.join(root, "img")
    if not os.path.isdir(imgs_dir):
        os.makedirs(root, exist_ok=True)
        # Common mirror: GitHub mirror often hosts Urban100.zip in SRBenchmark; but we don't hardcode here to avoid failure.
        # Require manual placement if auto-download not possible.
        # If user placed Urban100/*.png, accept that too.
        pngs = glob.glob(os.path.join(root, "*.png"))
        if pngs:
            return [np.array(Image.open(p).convert("RGB")) for p in sorted(pngs)]
        raise RuntimeError("Urban100 not found. Place PNGs under ./data/Urban100 or ./data/Urban100/img")
    files = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))
    if not files:
        raise RuntimeError("Urban100/img found but empty")
    return [np.array(Image.open(p).convert("RGB")) for p in files]

def load_images(dataset, root="./data"):
    ds = dataset.lower()
    if ds == "kodak24" or ds == "kodak":
        return load_kodak24(os.path.join(root,"kodak24"))
    if ds == "div2k":
        return load_div2k(os.path.join(root,"DIV2K"), subset="valid")
    if ds == "urban100":
        return load_urban100(os.path.join(root,"Urban100"))
    raise ValueError("Unknown dataset: " + dataset)
