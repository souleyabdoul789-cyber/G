"""
G
----------------------------------------------------------------
Lib d'entrainement de reseaux de neurones, coeur en C pur (portable
ARM32/ARM64/x86, sans PyTorch/TensorFlow), pilotee en Python.

Usage rapide :

    from G import Tensor, nn

    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    model = nn.Sequential([
        nn.Dense(4, 8, activation="relu"),
        nn.Dense(8, 1, activation="sigmoid"),
    ])
    y = model(x)
    y.backward()
"""

from .tensor import Tensor
from . import nn
from . import rl

__all__ = ["Tensor", "nn", "rl"]
__version__ = "0.1.0"
