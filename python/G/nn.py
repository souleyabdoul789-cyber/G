"""
G.nn
----------------------------------------------------------------
Couches de reseau de neurones et optimiseur, construits sur le
moteur de tenseurs C (voir G.tensor).
"""

import ctypes
from .tensor import Tensor, _lib, _PTensor

# ---------------------------------------------------------------
# Activation (doit correspondre a l'enum Activation dans nn.h)
# ---------------------------------------------------------------
ACT_NONE = 0
ACT_RELU = 1
ACT_SIGMOID = 2
ACT_TANH = 3


class _CDense(ctypes.Structure):
    _fields_ = [
        ("weights", _PTensor),
        ("bias", _PTensor),
        ("activation", ctypes.c_int),
        ("in_features", ctypes.c_int),
        ("out_features", ctypes.c_int),
    ]


_PDense = ctypes.POINTER(_CDense)

_lib.dense_create.restype = _PDense
_lib.dense_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.dense_free.argtypes = [_PDense]
_lib.dense_forward.restype = _PTensor
_lib.dense_forward.argtypes = [_PDense, _PTensor]

_lib.loss_mse.restype = _PTensor
_lib.loss_mse.argtypes = [_PTensor, _PTensor]


class Dense:
    """Couche entierement connectee : y = activation(x . W + b)

    Exemple :
        layer = Dense(4, 8, activation="relu")
        out = layer(x)   # x doit etre un Tensor (batch x 4)
    """

    _ACTS = {"none": ACT_NONE, "relu": ACT_RELU, "sigmoid": ACT_SIGMOID, "tanh": ACT_TANH}

    def __init__(self, in_features, out_features, activation="none", seed=42):
        if activation not in self._ACTS:
            raise ValueError(f"activation doit etre l'une de {list(self._ACTS)}")
        self._ptr = _lib.dense_create(in_features, out_features, self._ACTS[activation], seed)
        self.in_features = in_features
        self.out_features = out_features

    @property
    def weights(self):
        return Tensor(_ptr=self._ptr.contents.weights)

    @property
    def bias(self):
        return Tensor(_ptr=self._ptr.contents.bias)

    def parameters(self):
        return [self.weights, self.bias]

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(_ptr=_lib.dense_forward(self._ptr, x._ptr))

    def free(self):
        _lib.dense_free(self._ptr)


class Sequential:
    """Empile plusieurs couches. Exemple :

        model = Sequential([
            Dense(4, 16, activation="relu"),
            Dense(16, 1, activation="sigmoid"),
        ])
        out = model(x)
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def state_dict(self):
        """Renvoie les poids/biais de chaque couche, sous forme de listes Python
        (serialisable en JSON)."""
        return [{"weights": layer.weights.data, "bias": layer.bias.data} for layer in self.layers]

    def load_state_dict(self, state):
        """Recharge des poids/biais precedemment obtenus via state_dict().
        L'architecture (nombre de couches, tailles) doit deja correspondre."""
        for layer, s in zip(self.layers, state):
            layer.weights.set_data(s["weights"])
            layer.bias.set_data(s["bias"])

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    def load(self, path):
        import json
        with open(path) as f:
            state = json.load(f)
        self.load_state_dict(state)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return Tensor(_ptr=_lib.loss_mse(pred._ptr, target._ptr))


# ---------------------------------------------------------------
# SGD
# ---------------------------------------------------------------

class _CSGD(ctypes.Structure):
    _fields_ = [
        ("params", ctypes.POINTER(_PTensor)),
        ("n_params", ctypes.c_int),
        ("lr", ctypes.c_float),
    ]


_PSGD = ctypes.POINTER(_CSGD)

_lib.sgd_create.restype = _PSGD
_lib.sgd_create.argtypes = [ctypes.c_float]
_lib.sgd_add_param.argtypes = [_PSGD, _PTensor]
_lib.sgd_step.argtypes = [_PSGD]
_lib.sgd_zero_grad.argtypes = [_PSGD]
_lib.sgd_free.argtypes = [_PSGD]


class SGD:
    """Optimiseur simple : param -= lr * param.grad pour chaque parametre enregistre."""

    def __init__(self, parameters, lr=0.01):
        self._ptr = _lib.sgd_create(ctypes.c_float(lr))
        for p in parameters:
            _lib.sgd_add_param(self._ptr, p._ptr)

    def step(self):
        _lib.sgd_step(self._ptr)

    def zero_grad(self):
        _lib.sgd_zero_grad(self._ptr)

    def free(self):
        _lib.sgd_free(self._ptr)
