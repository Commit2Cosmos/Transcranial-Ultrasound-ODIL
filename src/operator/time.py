from base import SparseOperator, DenseOperator
import scipy.sparse as sp
import torch


class SparseTimeOperator(SparseOperator):
    """Time operator using central difference scheme"""

    def __init__(self, grid) -> None:
        super().__init__(grid)
        self.assemble()

    def _stencil(self) -> sp.csr_matrix:
        """Returns Dt: the central difference stencil for u_tt"""
        n = self.grid.nt
        dt = self.grid.dt
        return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / dt**2

    def assemble(self):
        """Assembles the time operator: Dt otimes Ixy"""
        Dt = self._stencil()
        Ixy = sp.eye(self.grid.nx * self.grid.ny)
        self.operator = sp.kron(Dt, Ixy)


class DenseTimeOperator(DenseOperator):
    """Time operator using central difference scheme"""

    def __init__(self, grid) -> None:
        super().__init__(grid)

    def gradient(self, u) -> torch.Tensor:
        """Computes the gradient w.r.t u using .roll"""
        d2udt2 = (
            torch.roll(u, -1, dims=0) - 2 * u + torch.roll(u, 1, dims=0)
        ) / self.grid.dt**2
        return d2udt2
