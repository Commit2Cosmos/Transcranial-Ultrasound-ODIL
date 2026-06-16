from abc import ABC, abstractmethod
import scipy.sparse as sp
import torch


class SparseOperator(ABC):
    """Base class for sparse operators used in direct solves"""

    def __init__(self, grid) -> None:
        self.grid = grid

    @abstractmethod
    def _stencil(self) -> sp.csr_matrix:
        pass

    @abstractmethod
    def assemble(self) -> None:
        pass


class DenseOperator(ABC):
    """Base class for dense operators used for inverse problems"""

    def __init__(self, grid) -> None:
        self.grid = grid

    @abstractmethod
    def gradient(self, u) -> torch.Tensor:
        """Computes the gradient w.r.t u using .roll"""
        pass
