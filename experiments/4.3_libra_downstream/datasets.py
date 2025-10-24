import torchvision as tv
import torchvision.transforms as T

def get_dataset(name, root="./data", train=True):
    name = name.lower()
    if name == "mnist":
        tfm = T.Compose([T.ToTensor()])
        return tv.datasets.MNIST(root, train=train, download=True, transform=tfm), 28*28, 10
    if name == "emnistb":
        tfm = T.Compose([T.ToTensor()])
        return tv.datasets.EMNIST(root, split="balanced", train=train, download=True, transform=tfm), 28*28, 47
    if name == "emnistl":
        tfm = T.Compose([T.ToTensor()])
        return tv.datasets.EMNIST(root, split="letters", train=train, download=True, transform=tfm), 28*28, 26
    if name == "fmnist":
        tfm = T.Compose([T.ToTensor()])
        return tv.datasets.FashionMNIST(root, train=train, download=True, transform=tfm), 28*28, 10
    if name == "kmnist":
        tfm = T.Compose([T.ToTensor()])
        return tv.datasets.KMNIST(root, train=train, download=True, transform=tfm), 28*28, 10
    if name == "cifar10":
        tfm = T.Compose([T.ToTensor()])
        return tv.datasets.CIFAR10(root, train=train, download=True, transform=tfm), 3*32*32, 10
    if name == "cifar100":
        tfm = T.Compose([T.ToTensor()])
        return tv.datasets.CIFAR100(root, train=train, download=True, transform=tfm), 3*32*32, 100
    if name == "svhn":
        tfm = T.Compose([T.ToTensor()])
        split = "train" if train else "test"
        return tv.datasets.SVHN(root, split=split, download=True, transform=tfm), 3*32*32, 10
    raise ValueError("Unknown dataset: " + name)
