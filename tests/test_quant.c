#include <stdint.h>

#include "../src/matrix.h"
#include "../src/quant.h"
#include "check.h"

static void test_round_trip_error(void) {
    float x[128], back[128], scales[4];
    int8_t q[128];
    for (int i = 0; i < 128; i++) x[i] = random_float() * 3.0f;

    quantize_row(x, 128, q, scales);
    dequantize_row(q, scales, 128, back);
    for (int i = 0; i < 128; i++) {
        CHECK_NEAR(back[i], x[i], scales[i / QK] * 0.5 + 1e-6);
    }
}

static void test_known_values(void) {
    float x[QK] = {0};
    float scales[1];
    int8_t q[QK];
    x[0] = 2.0f;
    x[1] = -2.0f;
    x[2] = 1.0f;

    quantize_row(x, QK, q, scales);
    CHECK_NEAR(scales[0], 2.0 / 127.0, 1e-7);
    CHECK(q[0] == 127);
    CHECK(q[1] == -127);
    CHECK(q[2] == 64);
    CHECK(q[3] == 0);
}

static void test_zero_block(void) {
    float x[QK] = {0};
    float scales[1];
    int8_t q[QK];
    for (int i = 0; i < QK; i++) q[i] = 55;

    quantize_row(x, QK, q, scales);
    CHECK(scales[0] == 0.0f);
    for (int i = 0; i < QK; i++) CHECK(q[i] == 0);
}

static void test_padding(void) {
    /* 40 values need two blocks; the 24 values after the end must be zero */
    float x[40];
    float scales[2];
    int8_t q[64];
    for (int i = 0; i < 40; i++) x[i] = random_float();

    CHECK(quant_blocks(40) == 2);
    quantize_row(x, 40, q, scales);
    for (int i = 40; i < 64; i++) CHECK(q[i] == 0);
}

static void test_simd_matches_scalar(void) {
    int8_t a[QK], b[QK];

    for (int trial = 0; trial < 20000; trial++) {
        for (int i = 0; i < QK; i++) {
            a[i] = (int8_t)(random_float() * 127.0f);
            b[i] = (int8_t)(random_float() * 127.0f);
        }
        CHECK(dot_q8(a, b) == dot_q8_scalar(a, b));
    }

    /* the quantizer never produces -128, so 127 and -127 are the extremes */
    const int8_t edge[4] = {127, -127, 0, 1};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < QK; k++) {
                a[k] = edge[i];
                b[k] = edge[j];
            }
            CHECK(dot_q8(a, b) == dot_q8_scalar(a, b));
        }
    }
}

static void test_matvec_close_to_f32(void) {
    enum { ROWS = 20, COLS = 50 };
    float w[ROWS * COLS], x[COLS], expected[ROWS], got[ROWS];
    for (int i = 0; i < ROWS * COLS; i++) w[i] = random_float();
    for (int i = 0; i < COLS; i++) x[i] = random_float();

    Matrix m = { w, ROWS, COLS };
    matvec_f32(expected, &m, x);

    QMatrix qm = qmatrix_from_f32(&m);
    int8_t xq[2 * QK];
    float xscales[2];
    quantize_row(x, COLS, xq, xscales);
    matvec_q8(got, &qm, xq, xscales);

    double largest = 0.0, worst = 0.0;
    for (int r = 0; r < ROWS; r++) {
        double e = fabs((double)got[r] - (double)expected[r]);
        if (e > worst) worst = e;
        if (fabs((double)expected[r]) > largest) largest = fabs((double)expected[r]);
    }
    printf("  matvec_q8: worst error %.4f, largest output %.3f\n", worst, largest);
    CHECK(worst < 0.05 * largest);
    qmatrix_free(&qm);
}

int main(void) {
    printf("kernel: %s\n", q8_kernel_name());
    test_round_trip_error();
    test_known_values();
    test_zero_block();
    test_padding();
    test_simd_matches_scalar();
    test_matvec_close_to_f32();
    return finish("test_quant");
}
