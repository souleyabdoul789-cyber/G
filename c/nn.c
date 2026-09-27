#include <stdlib.h>
#include "nn.h"

/* ============================================================
 * Dense layer
 * ============================================================ */

Dense *dense_create(int in_features, int out_features, Activation activation, unsigned int seed) {
    Dense *layer = (Dense *)malloc(sizeof(Dense));
    layer->weights = tensor_randn(in_features, out_features, 1, seed);
    layer->bias = tensor_zeros(1, out_features, 1);
    layer->activation = activation;
    layer->in_features = in_features;
    layer->out_features = out_features;
    return layer;
}

void dense_free(Dense *layer) {
    if (!layer) return;
    tensor_free(layer->weights);
    tensor_free(layer->bias);
    free(layer);
}

Tensor *dense_forward(Dense *layer, Tensor *x) {
    Tensor *z = tensor_matmul(x, layer->weights);
    if (!z) return NULL;
    Tensor *z_biased = tensor_add(z, layer->bias);
    if (!z_biased) return NULL;

    switch (layer->activation) {
        case ACT_RELU:    return tensor_relu(z_biased);
        case ACT_SIGMOID: return tensor_sigmoid(z_biased);
        case ACT_TANH:    return tensor_tanh(z_biased);
        case ACT_NONE:
        default:          return z_biased;
    }
}

/* ============================================================
 * SGD optimizer
 * ============================================================ */

SGD *sgd_create(float lr) {
    SGD *opt = (SGD *)malloc(sizeof(SGD));
    opt->params = NULL;
    opt->n_params = 0;
    opt->lr = lr;
    return opt;
}

void sgd_free(SGD *opt) {
    if (!opt) return;
    free(opt->params); /* does NOT free the tensors themselves, they belong to layers */
    free(opt);
}

void sgd_add_param(SGD *opt, Tensor *param) {
    opt->params = (Tensor **)realloc(opt->params, sizeof(Tensor *) * (opt->n_params + 1));
    opt->params[opt->n_params++] = param;
}

void sgd_step(SGD *opt) {
    for (int p = 0; p < opt->n_params; p++) {
        Tensor *t = opt->params[p];
        if (!t->requires_grad || !t->grad) continue;
        for (int i = 0; i < t->rows * t->cols; i++)
            t->data[i] -= opt->lr * t->grad[i];
    }
}

void sgd_zero_grad(SGD *opt) {
    for (int p = 0; p < opt->n_params; p++) {
        Tensor *t = opt->params[p];
        if (t->requires_grad && t->grad)
            for (int i = 0; i < t->rows * t->cols; i++) t->grad[i] = 0.0f;
    }
}

/* ============================================================
 * Loss functions
 * ============================================================ */

Tensor *loss_mse(Tensor *pred, Tensor *target) {
    Tensor *diff = tensor_sub(pred, target);
    if (!diff) return NULL;
    Tensor *sq = tensor_mul(diff, diff);
    Tensor *total = tensor_sum(sq);
    int n = pred->rows * pred->cols;
    Tensor *mean = tensor_scale(total, 1.0f / (float)n);
    return mean;
}
