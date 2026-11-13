#include <math.h>
#include <stdio.h>

#include "../src/matrix.h"

int main(void) {
    const float w[12] = {0.5f, -1.0f, 2.0f, 0.1f,
                         0.0f, 0.3f, -0.7f, 1.2f,
                         -2.0f, 0.9f, 0.4f, 0.0f};
    const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float expected[3] = {4.90f, 3.30f, 1.00f};

    Matrix m = {w, 3, 4};
    float out[3];
    matvec_f32(out, &m, x);

    int failed = 0;
    for (int r = 0; r < 3; r++) {
        if (fabsf(out[r] - expected[r]) > 1e-5f) {
            printf("FAIL: out[%d] = %f, expected %f\n", r, out[r], expected[r]);
            failed = 1;
        }
    }
    if (!failed)
        printf("PASS: matvec_f32\n");
    return failed;
}
