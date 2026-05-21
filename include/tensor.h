#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    float *data;
    int    channels;
    int    height;
    int    width;
    size_t size;   /* total number of floats */
} Tensor;

Tensor *tensor_create(int channels, int height, int width);
void    tensor_free(Tensor *t);
void    tensor_fill_random(Tensor *t);
void    tensor_fill(Tensor *t, float val);

#endif /* TENSOR_H */