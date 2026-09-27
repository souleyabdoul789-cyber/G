"""
G.tensor
----------------------------------------------------------------
Binding Python (ctypes) autour du moteur C de G.

Ce module ne reimplemente aucun calcul : chaque operation appelle
directement la lib C compilee (libg.so). C'est ce qui rend
G utilisable sur des machines contraintes (telephone ARM
via Termux) sans dependre de PyTorch/TensorFlow.
"""

import ctypes
import os

# ---------------------------------------------------------------
# Chargement de la librairie partagee
# ---------------------------------------------------------------
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libg.so")
_lib = ctypes.CDLL(_LIB_PATH)


class _CTensor(ctypes.Structure):
    """Miroir de la struct Tensor definie dans c/tensor.h.
    L'ordre et les types DOIVENT correspondre exactement au C."""
    pass


_CTensor._fields_ = [
    ("data", ctypes.POINTER(ctypes.c_float)),
    ("grad", ctypes.POINTER(ctypes.c_float)),
    ("rows", ctypes.c_int),
    ("cols", ctypes.c_int),
    ("requires_grad", ctypes.c_int),
    ("op", ctypes.c_int),
    ("parent_a", ctypes.POINTER(_CTensor)),
    ("parent_b", ctypes.POINTER(_CTensor)),
    ("visited", ctypes.c_int),
    ("scalar", ctypes.c_float),
]

_PTensor = ctypes.POINTER(_CTensor)

# ---------------------------------------------------------------
# Signatures des fonctions C (obligatoire pour eviter les crashs
# silencieux : ctypes suppose "int" par defaut sinon)
# ---------------------------------------------------------------
_lib.tensor_zeros.restype = _PTensor
_lib.tensor_zeros.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.tensor_from_array.restype = _PTensor
_lib.tensor_from_array.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.tensor_randn.restype = _PTensor
_lib.tensor_randn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]

_lib.tensor_free.argtypes = [_PTensor]
_lib.tensor_free_graph.argtypes = [_PTensor]

for name in ("tensor_add", "tensor_sub", "tensor_mul", "tensor_matmul"):
    fn = getattr(_lib, name)
    fn.restype = _PTensor
    fn.argtypes = [_PTensor, _PTensor]

for name in ("tensor_relu", "tensor_sigmoid", "tensor_tanh", "tensor_sum"):
    fn = getattr(_lib, name)
    fn.restype = _PTensor
    fn.argtypes = [_PTensor]

_lib.tensor_scale.restype = _PTensor
_lib.tensor_scale.argtypes = [_PTensor, ctypes.c_float]

_lib.tensor_backward.argtypes = [_PTensor]
_lib.tensor_zero_grad.argtypes = [_PTensor]

_lib.tensor_print.argtypes = [_PTensor, ctypes.c_char_p]


class Tensor:
    """Wrapper Python confortable autour du Tensor C.

    Exemple :
        a = Tensor([[1.0, 2.0]])
        b = Tensor([[3.0, 4.0]], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        print(b.grad)  # gradient de c par rapport a b
    """

    def __init__(self, values=None, rows=None, cols=None, requires_grad=False, _ptr=None):
        if _ptr is not None:
            self._ptr = _ptr
            return

        if values is not None:
            flat, r, c = _flatten(values)
            arr = (ctypes.c_float * len(flat))(*flat)
            self._ptr = _lib.tensor_from_array(arr, r, c, int(requires_grad))
        elif rows is not None and cols is not None:
            self._ptr = _lib.tensor_zeros(rows, cols, int(requires_grad))
        else:
            raise ValueError("Fournir soit `values`, soit `rows` et `cols`.")

    # -- proprietes --------------------------------------------------

    @property
    def shape(self):
        c = self._ptr.contents
        return (c.rows, c.cols)

    @property
    def data(self):
        c = self._ptr.contents
        n = c.rows * c.cols
        flat = [c.data[i] for i in range(n)]
        return _reshape(flat, c.rows, c.cols)

    @property
    def grad(self):
        c = self._ptr.contents
        if not c.grad:
            return None
        n = c.rows * c.cols
        flat = [c.grad[i] for i in range(n)]
        return _reshape(flat, c.rows, c.cols)

    # -- operations ----------------------------------------------------

    def __add__(self, other):
        return Tensor(_ptr=_lib.tensor_add(self._ptr, other._ptr))

    def __sub__(self, other):
        return Tensor(_ptr=_lib.tensor_sub(self._ptr, other._ptr))

    def __mul__(self, other):
        return Tensor(_ptr=_lib.tensor_mul(self._ptr, other._ptr))

    def matmul(self, other):
        return Tensor(_ptr=_lib.tensor_matmul(self._ptr, other._ptr))

    def relu(self):
        return Tensor(_ptr=_lib.tensor_relu(self._ptr))

    def sigmoid(self):
        return Tensor(_ptr=_lib.tensor_sigmoid(self._ptr))

    def tanh(self):
        return Tensor(_ptr=_lib.tensor_tanh(self._ptr))

    def sum(self):
        return Tensor(_ptr=_lib.tensor_sum(self._ptr))

    def scale(self, k):
        return Tensor(_ptr=_lib.tensor_scale(self._ptr, ctypes.c_float(k)))

    def backward(self):
        _lib.tensor_backward(self._ptr)

    def zero_grad(self):
        _lib.tensor_zero_grad(self._ptr)

    def free_graph(self):
        """Libere ce tenseur ET tout le sous-graphe qui l'a produit.
        Ne pas appeler sur des tenseurs feuilles que tu veux garder
        (poids d'un layer par exemple) : utilise `free()` pour ceux-la."""
        _lib.tensor_free_graph(self._ptr)

    def free(self):
        _lib.tensor_free(self._ptr)

    def set_data(self, values):
        """Ecrase les valeurs du tenseur en place (memoire C partagee).
        Utilise pour synchroniser un reseau cible (DQN) ou charger un
        modele sauvegarde. La forme de `values` doit correspondre exactement."""
        flat, r, c = _flatten(values)
        if (r, c) != self.shape:
            raise ValueError(f"set_data: forme incompatible {self.shape} vs {(r, c)}")
        arr = (ctypes.c_float * len(flat))(*flat)
        ctypes.memmove(self._ptr.contents.data, arr, ctypes.sizeof(arr))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"


@classmethod
def _randn(cls, rows, cols, requires_grad=False, seed=42):
    return cls(_ptr=_lib.tensor_randn(rows, cols, int(requires_grad), seed))


Tensor.randn = _randn


def _flatten(values):
    """Accepte une liste 1D ou 2D et renvoie (liste_plate, rows, cols)."""
    if isinstance(values[0], (list, tuple)):
        rows = len(values)
        cols = len(values[0])
        flat = [float(v) for row in values for v in row]
        return flat, rows, cols
    flat = [float(v) for v in values]
    return flat, 1, len(flat)


def _reshape(flat, rows, cols):
    if rows == 1 and cols == 1:
        return flat[0]
    if rows == 1:
        return flat
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]
