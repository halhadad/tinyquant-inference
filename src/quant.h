#ifndef TQ_QUANT_H
#define TQ_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "matrix.h"

/* Values are quantized in blocks of QK. Each block has its own scale. */
#define QK 32

/* int8 copy of a matrix. Each row is padded with zeros up to a multiple of QK,
 * so row r starts at q[r * blocks * QK] and its scales start at scales[r * blocks]. */
typedef struct {
    int8_t *q;
    float *scales;
    int rows;
    int cols;
    int blocks;
} QMatrix;

int quant_blocks(int n);

/* Writes quant_blocks(n) * QK values to q and quant_blocks(n) scales. */
void quantize_row(const float *x, int n, int8_t *q, float *scales);
void dequantize_row(const int8_t *q, const float *scales, int n, float *x);

QMatrix qmatrix_from_f32(const Matrix *m);
void qmatrix_free(QMatrix *m);
size_t qmatrix_bytes(const QMatrix *m);

/* Dot product of QK int8 pairs. dot_q8 uses AVX2 when it was compiled in.
 * The AVX2 version needs every value to be in -127..127. A -128 can overflow it.
 * quantize_row never produces -128. */
int32_t dot_q8_scalar(const int8_t *a, const int8_t *b);
int32_t dot_q8(const int8_t *a, const int8_t *b);
const char *q8_kernel_name(void);

/* out = m * x, where x was quantized with quantize_row. */
void matvec_q8(float *out, const QMatrix *m, const int8_t *xq, const float *xscales);

#endif
