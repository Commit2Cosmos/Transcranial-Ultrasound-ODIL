from abc import ABC, abstractmethod

import scipy.optimize as scopt
import numpy as np
from src.loss import DiscreteLoss, ForwardLoss, InverseLoss
from src.wavefield import Wavefield


class Optimiser(ABC):
    """Base optimiser class"""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        self.loss = loss  # loss function
        self.wavefield = wavefield  # wavefield to optimise

    @abstractmethod  # to be implemented by classes that inherit
    def minimise(self, u0, **kwargs):
        pass


class ScipyOptimiser(Optimiser):
    """Wrapper around scipy.optimize.minimize"""

    def __init__(
        self, wavefield: Wavefield, loss: DiscreteLoss, method: str = "L-BFGS-B", **opts
    ) -> None:
        super().__init__(wavefield, loss)
        self.method = method  # e.g., 'L-BFGS-B', 'Newton-CG'
        self.opts = opts  # e.g., maxiter, ftol

    def minimise(self, callback=None) -> scopt.OptimizeResult:
        # use amplitude data only for the forward
        if isinstance(self.loss, ForwardLoss):
            # tile amplitude for each shot, since forward loss only optimises amplitude
            u0 = np.tile(
                self.wavefield.amplitude.cpu().numpy().ravel(),
                self.loss.config.geometry.n_sources,
            )
        elif isinstance(self.loss, InverseLoss):
            n_shots = self.loss.config.geometry.n_sources  # extract shots
            amp0 = np.tile(
                self.wavefield.amplitude.cpu().numpy().ravel(), n_shots
            )  # tile amplitude for each shot
            wsp0 = (
                self.wavefield.wavespeed.cpu().numpy().ravel()
            )  # tile wavespeed for each shot
            u0 = np.concatenate(
                [amp0, wsp0]
            )  # concatenate amplitude and wavespeed for inverse problem

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
        self,
        wavefield: Wavefield,
        loss: DiscreteLoss,
        maxiter: int = 500,
        ftol: float = 1e-8,
        **opts,
    ) -> None:
        super().__init__(
            wavefield, loss, method="L-BFGS-B", maxiter=maxiter, ftol=ftol, **opts
        )
