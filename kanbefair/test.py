import torch
from models.KAN import build_kan_classifier

model = build_kan_classifier(input_shape=(1, 28, 28), num_classes=10, hidden=64, depth=2)
print(model)

x = torch.randn(8, 1, 28, 28)
out = model(x)
print(out.shape)