#ifndef TQ_UTIL_H
#define TQ_UTIL_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((noreturn))
static inline void die(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "tinyquant: ");
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
    exit(1);
}

static inline void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (p == NULL) die("out of memory (%zu bytes)", bytes);
    return p;
}

#endif
