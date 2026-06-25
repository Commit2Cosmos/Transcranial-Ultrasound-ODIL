from .base import SparseOperator, DenseOperator
from .conditions import (
    NeumannMirrorBC2nd,
    NeumannMirrorBC4th,
)
from .spatial import Laplacian2ndOrder, Laplacian4thOrder
from .temporal import TimeOperator2ndOrder, TimeOperator4thOrder
from .utils import WaveEquation

__all__ = [
    "SparseOperator",
    "DenseOperator",
    "Laplacian2ndOrder",
    "Laplacian4thOrder",
    "NeumannMirrorBC2nd",
    "NeumannMirrorBC4th",
    "TimeOperator2ndOrder",
    "TimeOperator4thOrder",
    "WaveEquation",
]
