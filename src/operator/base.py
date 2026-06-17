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

    @abstractmethod
    def apply(self, wavefield) -> torch.Tensor:
        pass


class DenseOperator(ABC):
    """Matrix-free operator using local stencils (e.g. ``torch.roll``)."""

    def __init__(self, grid) -> None:
        self.grid = grid

    @abstractmethod
    def apply(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the discrete operator to field u."""
        pass
