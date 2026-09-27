#ifndef GHOSTLIB_NN_H
#define GHOSTLIB_NN_H

#include "tensor.h"

/* ============================================================
 * Couche dense (fully-connected) : y = activation(x . W + b)
 * ============================================================ */

typedef enum {
    ACT_NONE,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_TANH
} Activation;

typedef struct {
    Tensor *weights; /* in_features x out_features, requires_grad=1 */
    Tensor *bias;    /* 1 x out_features, requires_grad=1 */
    Activation activation;
    int in_features;
    int out_features;
} Dense;

Dense *dense_create(int in_features, int out_features, Activation activation, unsigned int seed);
void dense_free(Dense *layer); /* frees weights/bias too */

/* Forward pass. Input x must be (batch x in_features). Returns a new Tensor
 * that is part of the autograd graph (call tensor_backward on the loss, then
 * read layer->weights->grad / layer->bias->grad). */
Tensor *dense_forward(Dense *layer, Tensor *x);

/* ============================================================
 * SGD optimizer
 * ============================================================ */

typedef struct {
    Tensor **params;   /* array of pointers to parameter tensors (weights/biases) */
    int n_params;
    float lr;
} SGD;

SGD *sgd_create(float lr);
void sgd_free(SGD *opt);
/* Registers a parameter tensor to be updated by this optimizer. */
void sgd_add_param(SGD *opt, Tensor *param);
/* Applies param.data -= lr * param.grad for every registered parameter. */
void sgd_step(SGD *opt);
/* Zeros the .grad of every registered parameter (call before each backward). */
void sgd_zero_grad(SGD *opt);

/* ============================================================
 * Pertes (loss functions) -- retournent un tenseur scalaire (1x1)
 * ============================================================ */

/* Mean Squared Error : mean((pred - target)^2) */
Tensor *loss_mse(Tensor *pred, Tensor *target);

#endif /* GHOSTLIB_NN_H */
