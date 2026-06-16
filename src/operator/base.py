from abc import ABC, abstractmethod
import scipy.sparse as sp
import jax

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
    """Matrix-free operator using local stencils (e.g. ``jnp.roll``)."""

    def __init__(self, grid) -> None:
        self.grid = grid

    @abstractmethod
    def apply(self, u: jax.Array, **kwargs) -> jax.Array:
        """Apply the discrete operator to field u."""
        pass
