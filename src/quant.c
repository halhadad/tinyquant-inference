#include "quant.h"

#include <math.h>
#include <string.h>

#include "util.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

#ifdef __AVX2__

const char *q8_kernel_name(void) {
    return "avx2";
}

int32_t dot_q8(const int8_t *a, const int8_t *b) {
    __m256i va = _mm256_loadu_si256((const __m256i *)a);
    __m256i vb = _mm256_loadu_si256((const __m256i *)b);

    /* maddubs wants one unsigned operand, so move the sign of a onto b */
    __m256i abs_a = _mm256_sign_epi8(va, va);
    __m256i signed_b = _mm256_sign_epi8(vb, va);

    __m256i pairs16 = _mm256_maddubs_epi16(abs_a, signed_b);
    __m256i sums32 = _mm256_madd_epi16(pairs16, _mm256_set1_epi16(1));

    __m128i lo = _mm256_castsi256_si128(sums32);
    __m128i hi = _mm256_extracti128_si256(sums32, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

#else

const char *q8_kernel_name(void) {
    return "scalar";
}

int32_t dot_q8(const int8_t *a, const int8_t *b) {
    return dot_q8_scalar(a, b);
}

#endif

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
