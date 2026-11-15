"""Writes the MNIST test set to data/mnist_test.bin for the C program.

Format (little endian): "TQD1", int32 count, int32 dim, count*dim pixel bytes,
count label bytes. Uses only the standard library.
"""
import struct

from mnist import fetch, read_images, read_labels

count, dim, images = read_images(fetch("t10k-images-idx3-ubyte.gz"))
label_count, labels = read_labels(fetch("t10k-labels-idx1-ubyte.gz"))
assert count == label_count

with open("data/mnist_test.bin", "wb") as f:
    f.write(b"TQD1")
    f.write(struct.pack("<ii", count, dim))
    f.write(images)
    f.write(labels)

print(f"wrote data/mnist_test.bin ({count} images of {dim} pixels)")
