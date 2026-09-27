#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"

/* ============================================================
 * Creation / destruction
 * ============================================================ */

static Tensor *tensor_alloc(int rows, int cols, int requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->rows = rows;
    t->cols = cols;
    t->requires_grad = requires_grad;
    t->op = OP_LEAF;
    t->parent_a = NULL;
    t->parent_b = NULL;
    t->visited = 0;

    int size = rows * cols;
    t->data = (float *)calloc(size, sizeof(float));
    t->grad = requires_grad ? (float *)calloc(size, sizeof(float)) : NULL;
    return t;
}

Tensor *tensor_zeros(int rows, int cols, int requires_grad) {
    return tensor_alloc(rows, cols, requires_grad);
}

Tensor *tensor_from_array(const float *values, int rows, int cols, int requires_grad) {
    Tensor *t = tensor_alloc(rows, cols, requires_grad);
    memcpy(t->data, values, sizeof(float) * rows * cols);
    return t;
}

Tensor *tensor_randn(int rows, int cols, int requires_grad, unsigned int seed) {
    Tensor *t = tensor_alloc(rows, cols, requires_grad);
    int fan_in = cols > 0 ? cols : 1;
    float scale = sqrtf(1.0f / (float)fan_in);
    unsigned int state = seed;
    for (int i = 0; i < rows * cols; i++) {
        /* rand_r-style xorshift, portable, no libc rand() global state */
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        float u = (float)(state % 100000) / 100000.0f; /* [0,1) */
        t->data[i] = (u * 2.0f - 1.0f) * scale;
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t);
}

/* Collects every node reachable from root (via parent_a/parent_b) into a
 * malloc'd array, in post-order (children before parents). Marks visited=1
 * on each node as it is added, to avoid duplicates in diamond graphs. */
static void build_topo(Tensor *node, Tensor ***order, int *count, int *capacity) {
    if (!node || node->visited) return;
    node->visited = 1;
    build_topo(node->parent_a, order, count, capacity);
    build_topo(node->parent_b, order, count, capacity);
    if (*count >= *capacity) {
        *capacity *= 2;
        *order = (Tensor **)realloc(*order, sizeof(Tensor *) * (*capacity));
    }
    (*order)[(*count)++] = node;
}

static Tensor **topo_sort(Tensor *root, int *out_count) {
    int capacity = 16;
    Tensor **order = (Tensor **)malloc(sizeof(Tensor *) * capacity);
    int count = 0;
    build_topo(root, &order, &count, &capacity);
    /* reset visited flags for future calls */
    for (int i = 0; i < count; i++) order[i]->visited = 0;
    *out_count = count;
    return order;
}

void tensor_free_graph(Tensor *t) {
    int count;
    Tensor **order = topo_sort(t, &count);
    for (int i = 0; i < count; i++) tensor_free(order[i]);
    free(order);
}

/* ============================================================
 * Forward ops
 * ============================================================ */

static Tensor *make_result(int rows, int cols, int rg_a, int rg_b, TensorOp op,
                            Tensor *a, Tensor *b) {
    int requires_grad = rg_a || rg_b;
    Tensor *out = tensor_alloc(rows, cols, requires_grad);
    out->op = op;
    out->parent_a = a;
    out->parent_b = b;
    return out;
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
    /* supports same-shape add, and broadcasting b as a (1 x cols) bias row */
    if (a->rows == b->rows && a->cols == b->cols) {
        Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_ADD, a, b);
        for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] + b->data[i];
        return out;
    }
    if (b->rows == 1 && b->cols == a->cols) {
        Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_ADD, a, b);
        for (int i = 0; i < a->rows; i++)
            for (int j = 0; j < a->cols; j++)
                out->data[i * a->cols + j] = a->data[i * a->cols + j] + b->data[j];
        return out;
    }
    fprintf(stderr, "G: tensor_add shape mismatch (%dx%d) + (%dx%d)\n",
            a->rows, a->cols, b->rows, b->cols);
    return NULL;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "G: tensor_sub shape mismatch (%dx%d) - (%dx%d)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_SUB, a, b);
    for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] - b->data[i];
    return out;
}

Tensor *tensor_mul(Tensor *a, Tensor *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "G: tensor_mul shape mismatch (%dx%d) * (%dx%d)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_MUL, a, b);
    for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] * b->data[i];
    return out;
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "G: tensor_matmul shape mismatch (%dx%d) x (%dx%d)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    Tensor *out = make_result(a->rows, b->cols, a->requires_grad, b->requires_grad, OP_MATMUL, a, b);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++)
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            out->data[i * b->cols + j] = sum;
        }
    }
    return out;
}

Tensor *tensor_relu(Tensor *a) {
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_RELU, a, NULL);
    for (int i = 0; i < a->rows * a->cols; i++)
        out->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
    return out;
}

Tensor *tensor_sigmoid(Tensor *a) {
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_SIGMOID, a, NULL);
    for (int i = 0; i < a->rows * a->cols; i++)
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    return out;
}

Tensor *tensor_tanh(Tensor *a) {
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_TANH, a, NULL);
    for (int i = 0; i < a->rows * a->cols; i++)
        out->data[i] = tanhf(a->data[i]);
    return out;
}

Tensor *tensor_sum(Tensor *a) {
    Tensor *out = make_result(1, 1, a->requires_grad, 0, OP_SUM, a, NULL);
    float s = 0.0f;
    for (int i = 0; i < a->rows * a->cols; i++) s += a->data[i];
    out->data[0] = s;
    return out;
}

Tensor *tensor_scale(Tensor *a, float scalar) {
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_SCALE, a, NULL);
    out->scalar = scalar;
    for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] * scalar;
    return out;
}

/* ============================================================
 * Autograd
 * ============================================================ */

void tensor_zero_grad(Tensor *t) {
    int count;
    Tensor **order = topo_sort(t, &count);
    for (int i = 0; i < count; i++) {
        Tensor *n = order[i];
        if (n->requires_grad && n->grad)
            memset(n->grad, 0, sizeof(float) * n->rows * n->cols);
    }
    free(order);
}

static void accumulate(Tensor *dst, int idx, float value) {
    if (dst && dst->requires_grad && dst->grad) dst->grad[idx] += value;
}

void tensor_backward(Tensor *t) {
    if (t->rows != 1 || t->cols != 1) {
        fprintf(stderr, "G: tensor_backward requires a scalar (1x1) tensor\n");
        return;
    }
    int count;
    Tensor **order = topo_sort(t, &count);

    if (!t->grad) t->grad = (float *)calloc(1, sizeof(float));
    t->grad[0] = 1.0f; /* dL/dL = 1 */

    for (int idx = count - 1; idx >= 0; idx--) {
        Tensor *n = order[idx];
        Tensor *a = n->parent_a;
        Tensor *b = n->parent_b;
        if (!n->grad) continue;

        switch (n->op) {
            case OP_LEAF:
                break;

            case OP_ADD:
                if (a->rows == n->rows && a->cols == n->cols) {
                    for (int i = 0; i < n->rows * n->cols; i++) accumulate(a, i, n->grad[i]);
                }
                if (b->rows == 1 && b->cols == n->cols && a->rows == n->rows) {
                    /* b was broadcast as a bias row: sum gradient over rows */
                    for (int j = 0; j < n->cols; j++) {
                        float s = 0.0f;
                        for (int i = 0; i < n->rows; i++) s += n->grad[i * n->cols + j];
                        accumulate(b, j, s);
                    }
                } else if (b->rows == n->rows && b->cols == n->cols) {
                    for (int i = 0; i < n->rows * n->cols; i++) accumulate(b, i, n->grad[i]);
                }
                break;

            case OP_SUB:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    accumulate(a, i, n->grad[i]);
                    accumulate(b, i, -n->grad[i]);
                }
                break;

            case OP_MUL:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    accumulate(a, i, n->grad[i] * b->data[i]);
                    accumulate(b, i, n->grad[i] * a->data[i]);
                }
                break;

            case OP_MATMUL:
                /* C = A(mxk) . B(kxn) ; dA = dC . B^T ; dB = A^T . dC */
                if (a->requires_grad) {
                    for (int i = 0; i < a->rows; i++)
                        for (int k = 0; k < a->cols; k++) {
                            float s = 0.0f;
                            for (int j = 0; j < n->cols; j++)
                                s += n->grad[i * n->cols + j] * b->data[k * b->cols + j];
                            accumulate(a, i * a->cols + k, s);
                        }
                }
                if (b->requires_grad) {
                    for (int k = 0; k < b->rows; k++)
                        for (int j = 0; j < b->cols; j++) {
                            float s = 0.0f;
                            for (int i = 0; i < a->rows; i++)
                                s += a->data[i * a->cols + k] * n->grad[i * n->cols + j];
                            accumulate(b, k * b->cols + j, s);
                        }
                }
                break;

            case OP_RELU:
                for (int i = 0; i < n->rows * n->cols; i++)
                    accumulate(a, i, n->data[i] > 0.0f ? n->grad[i] : 0.0f);
                break;

            case OP_SIGMOID:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    float s = n->data[i];
                    accumulate(a, i, n->grad[i] * s * (1.0f - s));
                }
                break;

            case OP_TANH:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    float th = n->data[i];
                    accumulate(a, i, n->grad[i] * (1.0f - th * th));
                }
                break;

            case OP_SUM:
                for (int i = 0; i < a->rows * a->cols; i++)
                    accumulate(a, i, n->grad[0]);
                break;

            case OP_SCALE:
                for (int i = 0; i < n->rows * n->cols; i++)
                    accumulate(a, i, n->grad[i] * n->scalar);
                break;
        }
    }
    free(order);
}

/* ============================================================
 * Utilities
 * ============================================================ */

void tensor_print(const Tensor *t, const char *label) {
    if (label) printf("%s ", label);
    printf("Tensor(%dx%d):\n", t->rows, t->cols);
    for (int i = 0; i < t->rows; i++) {
        printf("  [");
        for (int j = 0; j < t->cols; j++) {
            printf("%8.4f", t->data[i * t->cols + j]);
            if (j < t->cols - 1) printf(", ");
        }
        printf("]\n");
    }
}
