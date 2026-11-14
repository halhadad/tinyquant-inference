#include "dataset.h"

#include <stdio.h>
#include <string.h>

#include "util.h"

static void read_exact(FILE *f, void *dst, size_t bytes, const char *path) {
    if (fread(dst, 1, bytes, f) != bytes) die("%s: file ends too early", path);
}

void dataset_load(Dataset *ds, const char *path) {
    memset(ds, 0, sizeof *ds);

    FILE *f = fopen(path, "rb");
    if (f == NULL) die("cannot open %s", path);

    char magic[4];
    read_exact(f, magic, sizeof magic, path);
    if (memcmp(magic, "TQD1", 4) != 0) die("%s: not a dataset file", path);

    int32_t count, dim;
    read_exact(f, &count, sizeof count, path);
    read_exact(f, &dim, sizeof dim, path);
    if (count < 1 || dim < 1) die("%s: bad header", path);
    ds->count = count;
    ds->dim = dim;

    size_t n = (size_t)count * (size_t)dim;
    ds->images = xmalloc(n);
    ds->labels = xmalloc((size_t)count);
    read_exact(f, ds->images, n, path);
    read_exact(f, ds->labels, (size_t)count, path);
    fclose(f);
}

void dataset_free(Dataset *ds) {
    free(ds->images);
    free(ds->labels);
    memset(ds, 0, sizeof *ds);
}
