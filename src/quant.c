#include "quant.h"

#include <math.h>
#include <string.h>

#include "util.h"

int quant_blocks(int n) {
    return (n + QK - 1) / QK;
}

void quantize_row(const float *x, int n, int8_t *q, float *scales) {
    int blocks = quant_blocks(n);
    for (int b = 0; b < blocks; b++) {
        float chunk[QK] = {0};
        int len = n - b * QK;
        if (len > QK)
            len = QK;
        memcpy(chunk, x + b * QK, (size_t)len * sizeof(float));

        float maxabs = 0.0f;
        for (int i = 0; i < QK; i++) {
            float a = fabsf(chunk[i]);
            if (a > maxabs)
                maxabs = a;
        }

        float scale = maxabs / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        scales[b] = scale;
        for (int i = 0; i < QK; i++) {
            q[b * QK + i] = (int8_t)lrintf(chunk[i] * inv);
        }
    }
}

void dequantize_row(const int8_t *q, const float *scales, int n, float *x) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)q[i] * scales[i / QK];
    }
}

QMatrix qmatrix_from_f32(const Matrix *m) {
    QMatrix out;
    out.rows = m->rows;
    out.cols = m->cols;
    out.blocks = quant_blocks(m->cols);
    out.q = xmalloc((size_t)out.rows * out.blocks * QK);
    out.scales = xmalloc((size_t)out.rows * out.blocks * sizeof(float));
    for (int r = 0; r < out.rows; r++) {
        quantize_row(m->data + (size_t)r * m->cols, m->cols,
                     out.q + (size_t)r * out.blocks * QK,
                     out.scales + (size_t)r * out.blocks);
    }
    return out;
}

void qmatrix_free(QMatrix *m) {
    free(m->q);
    free(m->scales);
    m->q = NULL;
    m->scales = NULL;
}

size_t qmatrix_bytes(const QMatrix *m) {
    size_t n = (size_t)m->rows * m->blocks;
    return n * QK + n * sizeof(float);
}

int32_t dot_q8_scalar(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < QK; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

const char *q8_kernel_name(void) {
    return "scalar";
}

int32_t dot_q8(const int8_t *a, const int8_t *b) {
    return dot_q8_scalar(a, b);
}

void matvec_q8(float *out, const QMatrix *m, const int8_t *xq, const float *xscales) {
    for (int r = 0; r < m->rows; r++) {
        const int8_t *wq = m->q + (size_t)r * m->blocks * QK;
        const float *ws = m->scales + (size_t)r * m->blocks;
        float sum = 0.0f;
        for (int b = 0; b < m->blocks; b++) {
            sum += (float)dot_q8(wq + b * QK, xq + b * QK) * ws[b] * xscales[b];
        }
        out[r] = sum;
    }
}
