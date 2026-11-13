#include "matrix.h"

void matvec_f32(float *out, const Matrix *m, const float *x) {
    for (int r = 0; r < m->rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < m->cols; c++) {
            sum += m->data[r * m->cols + c] * x[c];
        }
        out[r] = sum;
    }
}
