from base import SparseOperator
import scipy.sparse as sp

class SparseLaplacian(SparseOperator):
    """2D Laplacian Operator"""

    def __init__(self, grid) -> None:
        super().__init__(grid)
        self.assemble()

    def _stencil(self, n: int, h: float) -> sp.csr_matrix:
        """Returns [Dx, Dy]: the central difference stencils for u_xx and u_yy"""
        D = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / h**2
        return D
    
    def assemble(self):
        """Assembles the 2D Laplacian operator: Iy \otimes Dx + Dy \otimes Ix"""
        # assemble the differentiation matrices Dx and Dy
        # Dx is u_xx, Dy is u_yy
        Dx = self._stencil(self.grid.nx, self.grid.dx)
        Dy = self._stencil(self.grid.ny, self.grid.dy)
        
        # construct the identity matrices for the Kronecker products
        # Iy is ny*ny, Ix is nx*nx
        Iy = sp.eye(self.grid.ny)
        Ix = sp.eye(self.grid.nx)

        # construct the operator as a Kronecker sum
        self.operator = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)