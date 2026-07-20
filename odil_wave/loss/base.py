from abc import ABC, abstractmethod

import torch

from .utils import LossConfig, LossTape


def mean_abs_sq(x: torch.Tensor, dim=None) -> torch.Tensor:
    """Real mean of |x|² (works for real or complex ``x``)."""
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
        """|r_pde|_rms / |src|_rms of the last evaluation."""
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
        """Return a torch scalar loss; autograd handles gradients."""
        raise NotImplementedError

    @abstractmethod
    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        """Vectorised residual on shot-batched inputs."""
        raise NotImplementedError
