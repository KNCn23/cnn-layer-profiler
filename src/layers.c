#include "layers.h"
#include "tensor.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ── Conv2D ─────────────────────────────────────────────────────────────── */
Tensor *layer_conv2d(const Tensor *in,
                     int out_channels, int kH, int kW,
                     int stride, int pad) {
    int in_H  = in->height,  in_W  = in->width,  in_C  = in->channels;
    int out_H = (in_H + 2*pad - kH) / stride + 1;
    int out_W = (in_W + 2*pad - kW) / stride + 1;

    Tensor *out = tensor_create(out_channels, out_H, out_W);
    if (!out) return NULL;

    /* Random kernel weights (Xavier-like) */
    float scale = sqrtf(2.0f / (float)(in_C * kH * kW));
    size_t k_size = (size_t)out_channels * in_C * kH * kW;
    float *kernel = malloc(k_size * sizeof(float));
    for (size_t i = 0; i < k_size; i++)
        kernel[i] = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * scale;

    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                float acc = 0.0f;
                for (int ic = 0; ic < in_C; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;
                            float v = 0.0f;
                            if (ih >= 0 && ih < in_H && iw >= 0 && iw < in_W)
                                v = tensor_at(in, ic, ih, iw);
                            size_t ki = (size_t)((oc*in_C + ic)*kH + kh)*kW + kw;
                            acc += v * kernel[ki];
                        }
                    }
                }
                tensor_set(out, oc, oh, ow, acc);
            }
        }
    }
    free(kernel);
    return out;
}

/* ── ReLU ────────────────────────────────────────────────────────────────── */
Tensor *layer_relu(const Tensor *in) {
    Tensor *out = tensor_create(in->channels, in->height, in->width);
    if (!out) return NULL;
    for (size_t i = 0; i < in->size; i++)
        out->data[i] = in->data[i] > 0.0f ? in->data[i] : 0.0f;
    return out;
}

/* ── MaxPool ─────────────────────────────────────────────────────────────── */
Tensor *layer_maxpool(const Tensor *in, int kH, int kW, int stride) {
    int out_H = (in->height - kH) / stride + 1;
    int out_W = (in->width  - kW) / stride + 1;
    Tensor *out = tensor_create(in->channels, out_H, out_W);
    if (!out) return NULL;

    for (int c = 0; c < in->channels; c++) {
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                float mx = -1e30f;
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        float v = tensor_at(in, c,
                                            oh*stride+kh, ow*stride+kw);
                        if (v > mx) mx = v;
                    }
                tensor_set(out, c, oh, ow, mx);
            }
        }
    }
    return out;
}

/* ── Flatten ─────────────────────────────────────────────────────────────── */
Tensor *layer_flatten(const Tensor *in) {
    Tensor *out = tensor_create(1, 1, (int)in->size);
    if (!out) return NULL;
    memcpy(out->data, in->data, in->size * sizeof(float));
    return out;
}

/* ── Fully-Connected ─────────────────────────────────────────────────────── */
Tensor *layer_fc(const Tensor *in, int out_features) {
    int in_features = (int)in->size;
    Tensor *out = tensor_create(1, 1, out_features);
    if (!out) return NULL;

    float scale = sqrtf(2.0f / (float)in_features);
    float *W = malloc((size_t)out_features * in_features * sizeof(float));
    for (int i = 0; i < out_features * in_features; i++)
        W[i] = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * scale;

    for (int o = 0; o < out_features; o++) {
        float acc = 0.0f;
        for (int i = 0; i < in_features; i++)
            acc += in->data[i] * W[(size_t)o*in_features + i];
        out->data[o] = acc;
    }
    free(W);
    return out;
}

/* ── FLOPs helpers ───────────────────────────────────────────────────────── */
long long flops_conv2d(int in_C, int out_C, int kH, int kW, int out_H, int out_W) {
    return 2LL * in_C * kH * kW * out_C * out_H * out_W;
}
long long flops_fc(int in_features, int out_features) {
    return 2LL * in_features * out_features;
}
