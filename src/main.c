#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tensor.h"
#include "layers.h"
#include "profiler.h"

/* Macro: run a layer, record timing + memory, free input */
#define PROFILE_LAYER(name_str, expr, flop_count)          \
    do {                                                    \
        double _t0 = now_ms();                              \
        Tensor *_out = (expr);                              \
        double _t1 = now_ms();                              \
        if (!_out) { fprintf(stderr, "OOM\n"); exit(1); }  \
        profiles[np].time_ms      = _t1 - _t0;             \
        profiles[np].memory_bytes = tensor_bytes(_out);     \
        profiles[np].flops        = (flop_count);           \
        strncpy(profiles[np].name, (name_str), 63);        \
        np++;                                               \
        tensor_free(cur); cur = _out;                       \
    } while (0)

int main(void) {
    srand((unsigned)time(NULL));

    /* ── Input: 3-channel 224×224 image (like VGG/AlexNet input) ── */
    Tensor *cur = tensor_create(3, 224, 224);
    tensor_rand(cur);
    printf("Input  "); tensor_print_shape(cur);

    LayerProfile profiles[32];
    int np = 0;

    /* ── Block 1 ── */
    PROFILE_LAYER("Conv2D-1  (3→64,3×3)",
        layer_conv2d(cur, 64, 3, 3, 1, 1),
        flops_conv2d(3, 64, 3, 3, 224, 224));

    PROFILE_LAYER("ReLU-1",
        layer_relu(cur),
        0);

    PROFILE_LAYER("MaxPool-1 (2×2,s2)",
        layer_maxpool(cur, 2, 2, 2),
        0);

    /* ── Block 2 ── */
    PROFILE_LAYER("Conv2D-2  (64→128,3×3)",
        layer_conv2d(cur, 128, 3, 3, 1, 1),
        flops_conv2d(64, 128, 3, 3, 112, 112));

    PROFILE_LAYER("ReLU-2",
        layer_relu(cur),
        0);

    PROFILE_LAYER("MaxPool-2 (2×2,s2)",
        layer_maxpool(cur, 2, 2, 2),
        0);

    /* ── Block 3 ── */
    PROFILE_LAYER("Conv2D-3  (128→256,3×3)",
        layer_conv2d(cur, 256, 3, 3, 1, 1),
        flops_conv2d(128, 256, 3, 3, 56, 56));

    PROFILE_LAYER("ReLU-3",
        layer_relu(cur),
        0);

    PROFILE_LAYER("MaxPool-3 (2×2,s2)",
        layer_maxpool(cur, 2, 2, 2),
        0);

    /* ── Classifier ── */
    PROFILE_LAYER("Flatten",
        layer_flatten(cur),
        0);

    int flat = (int)cur->size;
    PROFILE_LAYER("FC-1 (→1024)",
        layer_fc(cur, 1024),
        flops_fc(flat, 1024));

    PROFILE_LAYER("FC-2 (→10)",
        layer_fc(cur, 10),
        flops_fc(1024, 10));

    tensor_free(cur);

    /* ── Report ── */
    profile_print(profiles, np);
    profile_export_csv(profiles, np, "profile_results.csv");

    return 0;
}
