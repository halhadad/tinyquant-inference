#ifndef TQ_CHECK_H
#define TQ_CHECK_H

#include <math.h>
#include <stdio.h>

static int checks_run = 0;
static int checks_failed = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        checks_run++;                                                    \
        if (!(cond)) {                                                   \
            checks_failed++;                                             \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
        }                                                                \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                            \
    do {                                                                 \
        double a_ = (a), b_ = (b);                                       \
        checks_run++;                                                    \
        if (fabs(a_ - b_) > (tol)) {                                     \
            checks_failed++;                                             \
            printf("  FAIL %s:%d: %s = %g, expected %g\n",               \
                   __FILE__, __LINE__, #a, a_, b_);                      \
        }                                                                \
    } while (0)

static inline int finish(const char *name) {
    printf("%s: %d checks, %d failed\n", name, checks_run, checks_failed);
    return checks_failed ? 1 : 0;
}

/* Same sequence every run, so a failing test can be reproduced. */
static unsigned int rng_state = 12345u;
static inline float random_float(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) / 8388608.0f - 1.0f;
}

#endif
