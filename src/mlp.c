#include "mlp.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "matrix.h"
#include "util.h"

static void read_exact(FILE *f, void *dst, size_t bytes, const char *path) {
    if (fread(dst, 1, bytes, f) != bytes) die("%s: file ends too early", path);
}

void mlp_load(Mlp *net, const char *path) {
    memset(net, 0, sizeof *net);

    FILE *f = fopen(path, "rb");
    if (f == NULL) die("cannot open %s", path);

    char magic[4];
    read_exact(f, magic, sizeof magic, path);
    if (memcmp(magic, "TQM1", 4) != 0) die("%s: not a model file", path);

    int32_t n_layers;
    read_exact(f, &n_layers, sizeof n_layers, path);
    if (n_layers < 1 || n_layers > MLP_MAX_LAYERS) die("%s: bad layer count %d", path, n_layers);
    net->n_layers = n_layers;

    for (int i = 0; i <= n_layers; i++) {
        int32_t size;
        read_exact(f, &size, sizeof size, path);
        if (size < 1 || size > MLP_MAX_WIDTH) die("%s: bad layer size %d", path, size);
        net->sizes[i] = size;
    }

    for (int l = 0; l < n_layers; l++) {
        size_t in = (size_t)net->sizes[l];
        size_t out = (size_t)net->sizes[l + 1];
        net->weights[l] = xmalloc(out * in * sizeof(float));
        net->bias[l] = xmalloc(out * sizeof(float));
        read_exact(f, net->weights[l], out * in * sizeof(float), path);
        read_exact(f, net->bias[l], out * sizeof(float), path);
    }
    fclose(f);
}

void mlp_free(Mlp *net) {
    for (int l = 0; l < net->n_layers; l++) {
        free(net->weights[l]);
        free(net->bias[l]);
        if (net->quantized) qmatrix_free(&net->qweights[l]);
    }
    memset(net, 0, sizeof *net);
}

void mlp_quantize(Mlp *net) {
    if (net->quantized) return;
    for (int l = 0; l < net->n_layers; l++) {
        Matrix m = { net->weights[l], net->sizes[l + 1], net->sizes[l] };
        net->qweights[l] = qmatrix_from_f32(&m);
    }
    net->quantized = 1;
}

static void add_bias(float *v, const float *bias, int n, int relu) {
    for (int i = 0; i < n; i++) {
        v[i] += bias[i];
        if (relu && v[i] < 0.0f) v[i] = 0.0f;
    }
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int mlp_predict_f32(const Mlp *net, const float *input) {
    float buf_a[MLP_MAX_WIDTH], buf_b[MLP_MAX_WIDTH];
    float *cur = buf_a;
    float *next = buf_b;
    memcpy(cur, input, (size_t)net->sizes[0] * sizeof(float));

    for (int l = 0; l < net->n_layers; l++) {
        Matrix m = { net->weights[l], net->sizes[l + 1], net->sizes[l] };
        matvec_f32(next, &m, cur);
        add_bias(next, net->bias[l], net->sizes[l + 1], l < net->n_layers - 1);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }
    return argmax(cur, net->sizes[net->n_layers]);
}

int mlp_predict_q8(const Mlp *net, const float *input) {
    if (!net->quantized) die("mlp_predict_q8: call mlp_quantize first");

    float buf_a[MLP_MAX_WIDTH], buf_b[MLP_MAX_WIDTH];
    int8_t xq[MLP_MAX_WIDTH];
    float xscales[MLP_MAX_WIDTH / QK];
    float *cur = buf_a;
    float *next = buf_b;
    memcpy(cur, input, (size_t)net->sizes[0] * sizeof(float));

    for (int l = 0; l < net->n_layers; l++) {
        quantize_row(cur, net->sizes[l], xq, xscales);
        matvec_q8(next, &net->qweights[l], xq, xscales);
        add_bias(next, net->bias[l], net->sizes[l + 1], l < net->n_layers - 1);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }
    return argmax(cur, net->sizes[net->n_layers]);
}

size_t mlp_bytes_f32(const Mlp *net) {
    size_t total = 0;
    for (int l = 0; l < net->n_layers; l++) {
        total += (size_t)net->sizes[l] * net->sizes[l + 1] * sizeof(float);
    }
    return total;
}

size_t mlp_bytes_q8(const Mlp *net) {
    size_t total = 0;
    for (int l = 0; l < net->n_layers; l++) {
        total += qmatrix_bytes(&net->qweights[l]);
    }
    return total;
}
