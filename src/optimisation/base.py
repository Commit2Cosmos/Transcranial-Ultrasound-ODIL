from abc import ABC, abstractmethod
from typing import Callable

# import torch
import scipy.optimize as scopt


class Optimiser(ABC):
    """Base optimiser class"""

    def __init__(self, loss_fn: Callable) -> None:
        self.loss_fn = loss_fn  # loss function
        self.grad_fn = None  # autograd(loss_fn)  # gradient provided by JAX autodiff

    @abstractmethod  # to be implemented by classes that inherit
    def minimise(self, u0, **kwargs):
        pass


class ScipyOptimiser(Optimiser):
    """Wrapper around scipy.optimize.minimize"""

    def __init__(self, loss_fn: Callable, method: str = "L-BFGS-B", **opts) -> None:
        super().__init__(loss_fn)
        self.method = method  # e.g., 'L-BFGS-B', 'Newton-CG'
        self.opts = opts  # e.g., maxiter, ftol

    def minimise(self, u0, callback=None) -> scopt.OptimizeResult:
        result = scopt.minimize(
            fun=self.loss_fn,
            x0=u0,
            method=self.method,
            jac=self.grad_fn,
            callback=callback,
            options=self.opts,
        )
        return result


class LBFGSB(ScipyOptimiser):
    """Subclass for L-BFGS method"""

    def __init__(
        self, loss_fn: Callable, maxiter: int = 500, ftol: float = 1e-8, **opts
    ) -> None:
        super().__init__(loss_fn, method="L-BFGS-B", maxiter=maxiter, ftol=ftol, **opts)


if __name__ == "__main__":

    def test_loss():
        return 0

    optimiser = LBFGSB(test_loss)
