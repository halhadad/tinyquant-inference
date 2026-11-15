import gzip
import os
import struct
import urllib.request

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DATA_DIR = "data"


def fetch(name):
    """Download one MNIST file into data/ if it isn't there yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print("downloading", name)
        urllib.request.urlretrieve(BASE_URL + name, path)
    return path


def read_images(path):
    """Returns (count, pixels_per_image, raw bytes)."""
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: not an MNIST image file")
        data = f.read()
    if len(data) != count * rows * cols:
        raise ValueError(f"{path}: unexpected size")
    return count, rows * cols, data


def read_labels(path):
    """Returns (count, raw bytes)."""
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: not an MNIST label file")
        data = f.read()
    if len(data) != count:
        raise ValueError(f"{path}: unexpected size")
    return count, data
