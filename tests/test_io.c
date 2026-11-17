#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../src/dataset.h"
#include "../src/mlp.h"
#include "check.h"

static const char *path = "test_io_tmp.bin";

/* Writes a 4 byte magic, some int32 values, then some raw bytes. */
static void write_file(const char *magic, const int32_t *ints, size_t n_ints,
                       const uint8_t *payload, size_t n_payload) {
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f == NULL) return;
    fwrite(magic, 1, 4, f);
    if (n_ints > 0) fwrite(ints, sizeof(int32_t), n_ints, f);
    if (n_payload > 0) fwrite(payload, 1, n_payload, f);
    fclose(f);
}

static void load_dataset(const char *p) {
    Dataset ds;
    dataset_load(&ds, p);
}

static void load_mlp(const char *p) {
    Mlp net;
    mlp_load(&net, p);
}

/* Runs a loader in a child process, since a bad file makes it exit. Returns 1 if the child
 * exited with an error. Its error message is thrown away. */
static int load_fails(void (*loader)(const char *), const char *p) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        if (freopen("/dev/null", "w", stderr) == NULL) _exit(0);
        loader(p);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

static void test_dataset(void) {
    const int32_t header[2] = {2, 3};
    const uint8_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    write_file("TQD1", header, 2, payload, 8);
    Dataset ds;
    dataset_load(&ds, path);
    CHECK(ds.count == 2 && ds.dim == 3);
    CHECK(ds.images[0] == 1 && ds.images[5] == 6);
    CHECK(ds.labels[0] == 7 && ds.labels[1] == 8);
    dataset_free(&ds);
    CHECK(ds.images == NULL && ds.count == 0);

    remove(path);
    CHECK(load_fails(load_dataset, path));

    write_file("TQM1", header, 2, payload, 8);
    CHECK(load_fails(load_dataset, path));

    write_file("TQD1", header, 2, payload, 3);
    CHECK(load_fails(load_dataset, path));

    const int32_t empty[2] = {0, 3};
    write_file("TQD1", empty, 2, payload, 0);
    CHECK(load_fails(load_dataset, path));
}

static void test_mlp(void) {
    remove(path);
    CHECK(load_fails(load_mlp, path));

    const int32_t good_header[4] = {2, 2, 2, 2};
    write_file("TQM2", good_header, 4, NULL, 0);
    CHECK(load_fails(load_mlp, path));

    /* a correct header, but the weights are missing */
    write_file("TQM1", good_header, 4, NULL, 0);
    CHECK(load_fails(load_mlp, path));

    const int32_t no_layers[1] = {0};
    write_file("TQM1", no_layers, 1, NULL, 0);
    CHECK(load_fails(load_mlp, path));

    const int32_t too_many_layers[1] = {MLP_MAX_LAYERS + 1};
    write_file("TQM1", too_many_layers, 1, NULL, 0);
    CHECK(load_fails(load_mlp, path));

    const int32_t zero_size[3] = {1, 0, 2};
    write_file("TQM1", zero_size, 3, NULL, 0);
    CHECK(load_fails(load_mlp, path));

    const int32_t huge_size[3] = {1, 2, MLP_MAX_WIDTH + 1};
    write_file("TQM1", huge_size, 3, NULL, 0);
    CHECK(load_fails(load_mlp, path));
}

int main(void) {
    test_dataset();
    test_mlp();
    remove(path);
    return finish("test_io");
}
