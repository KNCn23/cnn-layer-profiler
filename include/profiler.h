#ifndef PROFILER_H
#define PROFILER_H

#include "tensor.h"
#include <stddef.h>

typedef struct {
    char   name[64];
    double time_ms;
    size_t memory_bytes;
    long   flops;
} LayerProfile;

/* High-resolution wall-clock timer in milliseconds */
double now_ms(void);

void print_profile_table(const LayerProfile *p, int n);
void export_profile_csv(const LayerProfile *p, int n, const char *path);

#endif /* PROFILER_H */