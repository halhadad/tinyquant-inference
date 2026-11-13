#ifndef TQ_MLP_H
#define TQ_MLP_H

#include <stddef.h>

#include "quant.h"

#define MLP_MAX_LAYERS 8
#define MLP_MAX_WIDTH 1024

/* Fully connected network: layer l maps sizes[l] inputs to sizes[l + 1] outputs.
 * weights[l] is a (sizes[l + 1] x sizes[l]) row-major matrix. ReLU follows every
 * layer except the last. */
typedef struct {
    int n_layers;
    int sizes[MLP_MAX_LAYERS + 1];
    float *weights[MLP_MAX_LAYERS];
    float *bias[MLP_MAX_LAYERS];
    QMatrix qweights[MLP_MAX_LAYERS];
    int quantized;
} Mlp;

void mlp_load(Mlp *net, const char *path);
void mlp_free(Mlp *net);

/* Builds the int8 copies used by mlp_predict_q8. */
void mlp_quantize(Mlp *net);

/* Both return the index of the largest output. input has sizes[0] floats. */
int mlp_predict_f32(const Mlp *net, const float *input);
int mlp_predict_q8(const Mlp *net, const float *input);

size_t mlp_bytes_f32(const Mlp *net);
size_t mlp_bytes_q8(const Mlp *net);

#endif
