import torchvision as tv
import torchvision.transforms as T

def get_dataset(name, root="./data", train=True):
    name = name.lower()
    if name == "cifar10":
        tfm_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        tfm_te = T.Compose([T.ToTensor()])
        ds = tv.datasets.CIFAR10(root, train=train, download=True, transform=tfm_tr if train else tfm_te)
        return ds, 3*32*32, 10
    if name == "cifar100":
        tfm_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        tfm_te = T.Compose([T.ToTensor()])
        ds = tv.datasets.CIFAR100(root, train=train, download=True, transform=tfm_tr if train else tfm_te)
        return ds, 3*32*32, 100
    if name == "mnist":
        tfm = T.ToTensor()
        ds = tv.datasets.MNIST(root, train=train, download=True, transform=tfm)
        return ds, 28*28, 10
    raise ValueError("Unknown dataset: " + name)
