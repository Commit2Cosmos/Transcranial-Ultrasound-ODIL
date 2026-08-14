import torch

from .base import DiscreteLoss, mean_abs_sq


class ForwardLoss(DiscreteLoss):
    """Per-shot PDE-residual loss for the frequency-domain forward solve."""

    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        # amp, sources: (n_shots, nf, nx, ny) complex
        return self.config.wave_eq.residual(amp, wsp, self.sources)

    def evaluate(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        r = self._residuals(amp, wsp)
        pde_weight = self.config.weights["pde"]
        L = pde_weight * mean_abs_sq(r)  # global mean over shots, frequencies, space
        if L.is_complex():
            L = L.real

        self.evaluations += 1
        self._last_residuals = (r.detach(),)
        return L
