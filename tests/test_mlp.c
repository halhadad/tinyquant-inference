#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/mlp.h"
#include "check.h"

/* Writes a 2 -> 2 -> 2 network to a file in the format mlp_load expects. */
static void write_tiny_model(const char *path) {
    const float w0[4] = { 1, 0, 0, 1 };
    const float b0[2] = { 0, 0 };
    const float w1[4] = { 1, -1, -1, 1 };
    const float b1[2] = { 0, 0 };
    const int32_t header[4] = { 2, 2, 2, 2 };   /* n_layers, then three sizes */

    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f == NULL) return;
    fwrite("TQM1", 1, 4, f);
    fwrite(header, sizeof(int32_t), 4, f);
    fwrite(w0, sizeof(float), 4, f);
    fwrite(b0, sizeof(float), 2, f);
    fwrite(w1, sizeof(float), 4, f);
    fwrite(b1, sizeof(float), 2, f);
    fclose(f);
}

int main(void) {
    const char *path = "test_mlp_tiny.bin";
    write_tiny_model(path);

    Mlp net;
    mlp_load(&net, path);
    remove(path);

    CHECK(net.n_layers == 2);
    CHECK(net.sizes[0] == 2 && net.sizes[1] == 2 && net.sizes[2] == 2);
    mlp_quantize(&net);

    /* first layer keeps the input, second layer outputs (a - b, b - a) */
    const float first_bigger[2] = { 3.0f, 1.0f };
    const float second_bigger[2] = { 1.0f, 3.0f };
    CHECK(mlp_predict_f32(&net, first_bigger) == 0);
    CHECK(mlp_predict_f32(&net, second_bigger) == 1);
    CHECK(mlp_predict_q8(&net, first_bigger) == 0);
    CHECK(mlp_predict_q8(&net, second_bigger) == 1);

    /* ReLU on the hidden layer: negative inputs are cut to zero */
    const float negative[2] = { -5.0f, 2.0f };
    CHECK(mlp_predict_f32(&net, negative) == 1);
    CHECK(mlp_predict_q8(&net, negative) == 1);

    CHECK(mlp_bytes_f32(&net) == 2 * 2 * 4 * 2);

    mlp_free(&net);
    return finish("test_mlp");
}
