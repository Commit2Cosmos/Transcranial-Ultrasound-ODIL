from abc import ABC, abstractmethod

import scipy.optimize as scopt
from src.loss import DiscreteLoss


class Optimiser(ABC):
    """Base optimiser class"""

    def __init__(self, loss: DiscreteLoss) -> None:
        self.loss = loss  # loss function

    @abstractmethod  # to be implemented by classes that inherit
    def minimise(self, u0, **kwargs):
        pass


class ScipyOptimiser(Optimiser):
    """Wrapper around scipy.optimize.minimize"""

    def __init__(self, loss: DiscreteLoss, method: str = "L-BFGS-B", **opts) -> None:
        super().__init__(loss)
        self.method = method  # e.g., 'L-BFGS-B', 'Newton-CG'
        self.opts = opts  # e.g., maxiter, ftol

    def minimise(self, u0, callback=None) -> scopt.OptimizeResult:
        result = scopt.minimize(
            fun=self.loss.evaluate,
            x0=u0,
            method=self.method,
            jac=True,  # gradient provided by torch through .evaluate
            callback=callback,
            options=self.opts,
        )
        return result


class LBFGSB(ScipyOptimiser):
    """Subclass for L-BFGS method"""

    def __init__(
        self, loss: DiscreteLoss, maxiter: int = 500, ftol: float = 1e-8, **opts
    ) -> None:
        super().__init__(loss, method="L-BFGS-B", maxiter=maxiter, ftol=ftol, **opts)
