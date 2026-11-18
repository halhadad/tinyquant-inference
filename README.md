# tinyquant

A small C project for running neural network inference with int8 quantized weights.

It loads an MNIST classifier, runs it in fp32 and in int8, and compares accuracy,
speed and weight size. The int8 matrix-vector product has a plain C version and an
AVX2 version.

Only plain fully connected networks are supported: up to 8 layers, up to 1024 values
per layer, ReLU between layers, and the answer is the index of the largest output.
There are no convolutions, no attention and no other activations. The model file
format (`TQM1`) is specific to this project.

## How the quantization works

Weights are split into blocks of 32 values. For each block the largest absolute value
is found, `scale = max / 127`, and every value is stored as `round(value / scale)` in
an int8. The block's scale is stored next to it. The input of each layer is quantized
the same way at run time, so the inner loop is an int8 x int8 dot product that is
summed in int32 and multiplied by the two scales at the end.

## Build and test

    make          # builds ./tinyquant
    make test     # unit tests
    make asan     # the same tests under AddressSanitizer and UBSan

The Makefile uses `-march=native`, so the AVX2 code is used when the CPU has it.
`make ARCH=` builds without it, which uses the plain C fallback.
Binary files are read as little endian.

The AVX2 dot product needs int8 values in -127..127 (a -128 can overflow it). The
quantizer should never produce -128.

## Get a model and the data

The test data only needs the Python standard library:

    python3 tools/export_data.py

The model is trained with PyTorch. Install the dependencies first, ideally in a virtual
environment:

    pip install -r requirements.txt

(For a smaller CPU-only torch, install numpy normally and then run
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.)

Then train:

    python3 tools/train.py        # optional argument: number of epochs, default 8

This downloads MNIST into `data/` and writes `models/mnist_mlp.bin`.

## Run

    ./tinyquant models/mnist_mlp.bin data/mnist_test.bin

It prints accuracy, time and weight size for fp32 and int8, and how many images the
two modes classify differently.

## Results

Model: 784-512-256-10, trained for 8 epochs (`tools/train.py`). Machine: Intel Core
i5-8350U (AVX2) under WSL2. Best of 5 passes over the 10,000 MNIST test images.

| mode | accuracy | weights                      |
| ---- | -------- | ---------------------------- |
| fp32 | 98.07%   | 2,140,160 bytes              |
| int8 | 98.06%   | 611,136 bytes (3.5x smaller) |

The two modes disagree on 1 of 10,000 images. The fp32 accuracy matches what PyTorch
reports for the same model.

Time per image, over three runs:

| build              | fp32       | int8      |
| ------------------ | ---------- | --------- |
| default (`make`)   | 705-810 us | 70-115 us |
| with `-ffast-math` | 97-135 us  | 70-75 us  |

The default build compiles the fp32 loop to scalar code, because strict float rules
stop the compiler from reordering the sum. That makes the 6-10x gap in the first row
mostly a weak baseline. With `-ffast-math` the compiler can vectorize the fp32 loop, and
int8 is about 1.4-1.9x faster. Timings on my laptop under WSL2 vary a lot between runs.
A hand-written AVX2 fp32 kernel would be the fairest comparison.

## Layout

    src/matrix.*    matrix type and the fp32 matrix-vector product
    src/quant.*     block quantization, int8 dot product (scalar and AVX2), int8 matvec
    src/mlp.*       model loading and prediction (fp32 and int8)
    src/dataset.*   loads the test images
    src/main.c      the benchmark program
    tests/          unit tests
    tools/          Python scripts for the data and the model

## Notes

- The fp32 path is a plain C loop. Its speed depends on what the compiler does with it,
  so any speedup over fp32 is against that baseline and not against a hand-tuned
  fp32 kernel.

## License

MIT
