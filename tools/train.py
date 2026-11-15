"""Trains a small MNIST classifier and writes it to models/mnist_mlp.bin.

Needs numpy and torch. Run from the project root: python3 tools/train.py

Format (little endian): "TQM1", int32 n_layers, int32 sizes[n_layers + 1], then
for each layer the weights (out x in, row-major float32) and the bias (float32).
"""
import os
import struct
import sys

import numpy as np
import torch
from torch import nn

from mnist import fetch, read_images, read_labels

HIDDEN = [512, 256]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

torch.manual_seed(0)


def load(images_name, labels_name):
    count, dim, raw = read_images(fetch(images_name))
    _, labels = read_labels(fetch(labels_name))
    x = np.frombuffer(raw, dtype=np.uint8).reshape(count, dim).astype(np.float32) / 255.0
    y = np.frombuffer(labels, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def accuracy(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


train_x, train_y = load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
test_x, test_y = load("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

sizes = [train_x.shape[1]] + HIDDEN + [10]
layers = []
for i in range(len(sizes) - 1):
    layers.append(nn.Linear(sizes[i], sizes[i + 1]))
    if i < len(sizes) - 2:
        layers.append(nn.ReLU())
model = nn.Sequential(*layers)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    order = torch.randperm(train_x.shape[0])
    for start in range(0, len(order), 128):
        batch = order[start:start + 128]
        optimizer.zero_grad()
        loss = loss_fn(model(train_x[batch]), train_y[batch])
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}/{EPOCHS}: test accuracy {accuracy(model, test_x, test_y):.4f}")

os.makedirs("models", exist_ok=True)
linear = [m for m in model if isinstance(m, nn.Linear)]
with open("models/mnist_mlp.bin", "wb") as f:
    f.write(b"TQM1")
    f.write(struct.pack("<i", len(linear)))
    f.write(struct.pack(f"<{len(sizes)}i", *sizes))
    for layer in linear:
        f.write(layer.weight.detach().numpy().astype("<f4").tobytes())
        f.write(layer.bias.detach().numpy().astype("<f4").tobytes())

print("wrote models/mnist_mlp.bin")
