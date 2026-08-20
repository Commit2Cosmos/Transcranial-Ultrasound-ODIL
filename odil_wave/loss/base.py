from abc import ABC, abstractmethod

import torch

from .utils import LossConfig, LossTape


def mean_abs_sq(x: torch.Tensor, dim=None) -> torch.Tensor:
    """Real mean of ``|x|**2`` (works for real or complex ``x``).

    Parameters
    ----------
    x : torch.Tensor
        Real or complex tensor.
    dim : int or tuple of int, optional
        Dimensions to reduce; reduces over all elements when omitted.

    Returns
    -------
    torch.Tensor
        Real-valued mean of the squared magnitude.
    """
    if x.is_complex():
        val = x.real.square() + x.imag.square()
    else:
        val = x.square()
    if dim is None:
        return val.mean()
    return val.mean(dim=dim)


class DiscreteLoss(ABC):
    """Base class for discrete loss functions."""

    def __init__(
        self,
        config: LossConfig,
        callback: LossTape | None = None,
    ):
        """Initialise the loss and precompute the per-shot source fields.

        Parameters
        ----------
        config : LossConfig
            Wave-equation, geometry, weights and regulariser configuration.
        callback : LossTape, optional
            Diagnostics tape; a fresh :class:`LossTape` is created if omitted.
        """
        self.config = config
        self.callback = callback if callback is not None else LossTape()

        # Precompute complex source fields per shot; f' = t0**2 * f.
        t0 = self.config.wave_eq.wavefield.grid.t0
        cdtype = self.config.wave_eq.wavefield.cdtype
        self.sources = (
            torch.stack(
                [
                    self.config.geometry.source_field(i).to(
                        dtype=cdtype, device=self.config.device
                    )
                    for i in range(self.config.geometry.n_sources)
                ]
            )
            * t0**2
        )

        self.src_rms = float(mean_abs_sq(self.sources).sqrt())
        self.evaluations = 0

    def pde_src_ratio(self) -> float:
        """RMS PDE residual relative to the RMS source of the last evaluation.

        Returns
        -------
        float
            ``|r_pde|_rms / |src|_rms`` from the most recent evaluation.
        """
        residuals = self._last_residuals

        if isinstance(residuals, dict):
            if "pde_src_ratio" in residuals:
                return float(residuals["pde_src_ratio"])
            if "pde_rms" in residuals:
                return float(residuals["pde_rms"]) / max(self.src_rms, 1e-30)

        r_pde = residuals[0]
        return float(mean_abs_sq(r_pde.detach()).sqrt().cpu()) / max(
            self.src_rms, 1e-30
        )

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> torch.Tensor:
        """Return a torch scalar loss.

        Returns
        -------
        torch.Tensor
            Scalar loss; autograd handles gradients.
        """
        raise NotImplementedError

    @abstractmethod
    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        """Compute the vectorised residual on shot-batched inputs.

        Parameters
        ----------
        amp : torch.Tensor
            Wavefield amplitudes, shot-batched.
        wsp : torch.Tensor
            Wave speed / velocity field.

        Returns
        -------
        torch.Tensor
            Residual tensor(s) for the loss.
        """
        raise NotImplementedError
