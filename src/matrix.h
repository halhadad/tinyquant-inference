#ifndef TQ_MATRIX_H
#define TQ_MATRIX_H

/* Row-major matrix: element (r, c) is data[r * cols + c]. Doesn't own the memory. */
typedef struct {
    const float *data;
    int rows;
    int cols;
} Matrix;

/* out = m * x. x has m->cols floats, out has m->rows floats. */
void matvec_f32(float *out, const Matrix *m, const float *x);

#endif
