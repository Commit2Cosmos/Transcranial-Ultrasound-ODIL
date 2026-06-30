import torch

from .base import DiscreteLoss


class ForwardLoss(DiscreteLoss):
    """Per-shot PDE-residual loss for the forward solve."""

    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        # amp:     (n_shots, NT, NX, NY)
        # sources: (n_shots, NT, NX, NY) -- precomputed in DiscreteLoss.__init__
        return self.config.wave_eq.residual(amp, wsp, self.sources)

    def evaluate(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        r = self._residuals(amp, wsp)
        w_pde = self.config.weights["pde"]
        L = w_pde * (r**2).mean(dim=(1, 2, 3)).sum()

        self.evaluations += 1
        self._last_residuals = (r.detach(),)

        return L
