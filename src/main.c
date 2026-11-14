#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataset.h"
#include "mlp.h"
#include "quant.h"
#include "util.h"

#define PASSES 5

typedef int (*predict_fn)(const Mlp *, const float *);

typedef struct {
    double best_seconds;
    int correct;
    int *predictions;
} Result;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Runs every image PASSES times and keeps the fastest pass. */
static Result evaluate(const Mlp *net, predict_fn predict, const float *inputs, const Dataset *ds) {
    Result res;
    res.best_seconds = 1e30;
    res.correct = 0;
    res.predictions = xmalloc((size_t)ds->count * sizeof(int));

    for (int pass = 0; pass < PASSES; pass++) {
        double start = now_seconds();
        for (int i = 0; i < ds->count; i++) {
            res.predictions[i] = predict(net, inputs + (size_t)i * ds->dim);
        }
        double elapsed = now_seconds() - start;
        if (elapsed < res.best_seconds) res.best_seconds = elapsed;
    }

    for (int i = 0; i < ds->count; i++) {
        if (res.predictions[i] == ds->labels[i]) res.correct++;
    }
    return res;
}

static void print_row(const char *name, const Result *res, int count, size_t weight_bytes) {
    printf("%-6s %8.2f%% %10.1f ms %9.2f us %10zu\n", name,
           100.0 * res->correct / count,
           res->best_seconds * 1000.0,
           res->best_seconds * 1e6 / count,
           weight_bytes);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.bin> <test_data.bin>\n", argv[0]);
        return 1;
    }

    Mlp net;
    mlp_load(&net, argv[1]);
    Dataset ds;
    dataset_load(&ds, argv[2]);
    if (ds.dim != net.sizes[0]) {
        die("model expects %d inputs but the data has %d", net.sizes[0], ds.dim);
    }
    mlp_quantize(&net);

    size_t n = (size_t)ds.count * (size_t)ds.dim;
    float *inputs = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        inputs[i] = (float)ds.images[i] / 255.0f;
    }

    Result f32 = evaluate(&net, mlp_predict_f32, inputs, &ds);
    Result q8 = evaluate(&net, mlp_predict_q8, inputs, &ds);

    int differ = 0;
    for (int i = 0; i < ds.count; i++) {
        if (f32.predictions[i] != q8.predictions[i]) differ++;
    }

    printf("int8 kernel: %s, images: %d, passes: %d (fastest is shown)\n\n",
           q8_kernel_name(), ds.count, PASSES);
    printf("%-6s %9s %13s %12s %10s\n", "mode", "accuracy", "total time", "per image", "weights (B)");
    print_row("f32", &f32, ds.count, mlp_bytes_f32(&net));
    print_row("int8", &q8, ds.count, mlp_bytes_q8(&net));
    printf("\nf32 and int8 disagree on %d of %d images\n", differ, ds.count);

    free(f32.predictions);
    free(q8.predictions);
    free(inputs);
    dataset_free(&ds);
    mlp_free(&net);
    return 0;
}
