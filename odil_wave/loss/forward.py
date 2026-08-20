import torch

from .base import DiscreteLoss, mean_abs_sq


class ForwardLoss(DiscreteLoss):
    """Per-shot PDE-residual loss for the frequency-domain forward solve."""

    def _residuals(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        """Compute the PDE residual for the shot-batched wavefield.

        Parameters
        ----------
        amp : torch.Tensor
            Complex wavefield amplitudes ``(n_shots, nf, nx, ny)``.
        wsp : torch.Tensor
            Wave speed / velocity field.

        Returns
        -------
        torch.Tensor
            Complex PDE residual per shot.
        """
        return self.config.wave_eq.residual(amp, wsp, self.sources)

    def evaluate(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        """PDE-residual loss, mean-reduced over shots, frequencies and space.

        Parameters
        ----------
        amp : torch.Tensor
            Complex wavefield amplitudes ``(n_shots, nf, nx, ny)``.
        wsp : torch.Tensor
            Wave speed / velocity field.

        Returns
        -------
        torch.Tensor
            Real scalar loss ``pde_weight * mean(|r|**2)``.
        """
        r = self._residuals(amp, wsp)
        pde_weight = self.config.weights["pde"]
        L = pde_weight * mean_abs_sq(r)  # global mean over shots, frequencies, space
        if L.is_complex():
            L = L.real

        self.evaluations += 1
        self._last_residuals = (r.detach(),)
        return L
