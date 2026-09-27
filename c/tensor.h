#ifndef GHOSTLIB_TENSOR_H
#define GHOSTLIB_TENSOR_H

/*
 * G - tensor.h
 * ----------------------------------------------------------
 * Coeur du moteur de tenseurs + autograd, en C pur (aucune
 * dependance externe, portable sur ARM32/ARM64/x86).
 *
 * Un Tensor est un tableau 2D (matrice) de float32 :
 *   - rows x cols
 *   - data[i*cols + j] = element (i, j)
 *
 * Chaque Tensor peut faire partie d'un graphe de calcul :
 *   - si requires_grad == 1, il accumule son gradient dans `grad`
 *   - op / parent_a / parent_b decrivent comment il a ete produit
 *   - tensor_backward() remonte le graphe et calcule tous les
 *     gradients via la regle de la chaine (backprop manuelle).
 *
 * Ce n'est PAS un moteur generique a N dimensions : on reste en
 * 2D (vecteurs = matrices a 1 ligne ou 1 colonne) pour rester
 * simple, rapide et fiable sur du materiel contraint (telephone).
 */

typedef enum {
    OP_LEAF,      /* tenseur cree directement par l'utilisateur */
    OP_ADD,
    OP_SUB,
    OP_MUL,       /* multiplication elementwise */
    OP_MATMUL,    /* produit matriciel */
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_SUM,       /* reduction -> scalaire (tenseur 1x1) */
    OP_SCALE      /* multiplication par une constante float */
} TensorOp;

typedef struct Tensor {
    float *data;          /* rows * cols valeurs, row-major */
    float *grad;          /* meme taille que data, ou NULL si requires_grad == 0 */
    int rows;
    int cols;
    int requires_grad;

    /* graphe de calcul (pour l'autograd) */
    TensorOp op;
    struct Tensor *parent_a;
    struct Tensor *parent_b;   /* NULL si op unaire ou feuille */

    /* comptage de reference simplifie : on libere explicitement */
    int visited;          /* utilise en interne pour le tri topologique */
    float scalar;         /* utilise seulement par OP_SCALE */
} Tensor;

/* ---- creation / destruction ---- */

/* Cree un tenseur rempli de zeros. */
Tensor *tensor_zeros(int rows, int cols, int requires_grad);

/* Cree un tenseur a partir d'un tableau de floats fourni (copie les valeurs). */
Tensor *tensor_from_array(const float *values, int rows, int cols, int requires_grad);

/* Cree un tenseur avec des valeurs aleatoires (init "Xavier" simplifiee), utile pour les poids. */
Tensor *tensor_randn(int rows, int cols, int requires_grad, unsigned int seed);

/* Libere un seul tenseur (ne touche pas a ses parents). */
void tensor_free(Tensor *t);

/* Libere un tenseur ET tout le sous-graphe de calcul qui a servi a le produire
 * (utile en fin d'entrainement pour eviter les fuites memoire). Ne libere pas
 * les feuilles marquees comme parametres persistants (voir tensor_detach). */
void tensor_free_graph(Tensor *t);

/* ---- operations (construisent le graphe si un des operandes requires_grad) ---- */

Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);      /* elementwise */
Tensor *tensor_matmul(Tensor *a, Tensor *b);   /* produit matriciel classique */
Tensor *tensor_relu(Tensor *a);
Tensor *tensor_sigmoid(Tensor *a);
Tensor *tensor_tanh(Tensor *a);
Tensor *tensor_sum(Tensor *a);                 /* -> tenseur 1x1 */
Tensor *tensor_scale(Tensor *a, float scalar); /* a * scalar (constante non-apprenable) */

/* ---- autograd ---- */

/* Met tous les gradients du sous-graphe a zero. */
void tensor_zero_grad(Tensor *t);

/* Calcule dt/dparametres pour tout le graphe en amont de t.
 * t doit etre un scalaire (1x1), typiquement une loss. */
void tensor_backward(Tensor *t);

/* ---- utilitaires ---- */

void tensor_print(const Tensor *t, const char *label);

#endif /* GHOSTLIB_TENSOR_H */
