#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

/* All layer functions return a newly allocated Tensor.
   The caller is responsible for calling tensor_free() on both
   the input and the returned output when done. */

Tensor *layer_conv2d(const Tensor *in, int out_ch, int k, int stride, int pad);
Tensor *layer_relu(const Tensor *in);
Tensor *layer_maxpool(const Tensor *in, int k, int stride);
Tensor *layer_flatten(const Tensor *in);
Tensor *layer_fc(const Tensor *in, int out_features);

/* FLOPs helpers (returned as long) */
long flops_conv2d(const Tensor *in, int out_ch, int k);
long flops_fc(const Tensor *in, int out_features);

#endif /* LAYERS_H */