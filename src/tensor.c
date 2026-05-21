#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Tensor *tensor_create(int channels, int height, int width) {
    Tensor *t = malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->channels = channels;
    t->height   = height;
    t->width    = width;
    t->size     = (size_t)channels * height * width;
    t->data     = calloc(t->size, sizeof(float));
    if (!t->data) { free(t); return NULL; }
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

void tensor_fill(Tensor *t, float val) {
    for (size_t i = 0; i < t->size; i++) t->data[i] = val;
}

void tensor_rand(Tensor *t) {
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

float tensor_at(const Tensor *t, int c, int h, int w) {
    return t->data[(size_t)c * t->height * t->width + (size_t)h * t->width + w];
}

void tensor_set(Tensor *t, int c, int h, int w, float val) {
    t->data[(size_t)c * t->height * t->width + (size_t)h * t->width + w] = val;
}

size_t tensor_bytes(const Tensor *t) {
    return t->size * sizeof(float);
}

void tensor_print_shape(const Tensor *t) {
    printf("Tensor(%d x %d x %d, %.2f KB)\n",
           t->channels, t->height, t->width,
           (double)tensor_bytes(t) / 1024.0);
}
