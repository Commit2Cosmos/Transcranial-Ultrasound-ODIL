from abc import ABC, abstractmethod
import scipy.sparse as sp
import numpy as np

class Operator(ABC):
    """Base class for spatial operators"""
    def __init__(self, grid) -> None:
        self.grid = grid

    @abstractmethod
    def _stencil(self) -> sp.csr_matrix:
        pass

    @abstractmethod
    def assemble(self) -> None:
        pass