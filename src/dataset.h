#ifndef TQ_DATASET_H
#define TQ_DATASET_H

#include <stdint.h>

/* images is count * dim bytes (one byte per pixel), labels is count bytes. */
typedef struct {
    int count;
    int dim;
    uint8_t *images;
    uint8_t *labels;
} Dataset;

void dataset_load(Dataset *ds, const char *path);
void dataset_free(Dataset *ds);

#endif
