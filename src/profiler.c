#include "profiler.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void profile_print(const LayerProfile *profiles, int n) {
    printf("\n%-22s  %10s  %12s  %14s\n",
           "Layer", "Time(ms)", "Memory(KB)", "FLOPs");
    printf("%-22s  %10s  %12s  %14s\n",
           "---------------------", "--------", "----------", "-----");

    double total_ms  = 0;
    long long total_flops = 0;
    size_t total_mem = 0;

    for (int i = 0; i < n; i++) {
        printf("%-22s  %10.3f  %12.2f  %14lld\n",
               profiles[i].name,
               profiles[i].time_ms,
               profiles[i].memory_bytes / 1024.0,
               profiles[i].flops);
        total_ms    += profiles[i].time_ms;
        total_flops += profiles[i].flops;
        total_mem   += profiles[i].memory_bytes;
    }
    printf("%-22s  %10s  %12s  %14s\n",
           "---------------------", "--------", "----------", "-----");
    printf("%-22s  %10.3f  %12.2f  %14lld\n",
           "TOTAL", total_ms, total_mem / 1024.0, total_flops);
    printf("\nThroughput: %.2f MFLOP/s\n\n",
           total_flops / (total_ms * 1e3));
}

void profile_export_csv(const LayerProfile *profiles, int n, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); return; }
    fprintf(f, "layer,time_ms,memory_bytes,flops\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%s,%.6f,%zu,%lld\n",
                profiles[i].name,
                profiles[i].time_ms,
                profiles[i].memory_bytes,
                profiles[i].flops);
    fclose(f);
    printf("Profile exported to %s\n", path);
}
