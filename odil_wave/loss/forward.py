import torch

from .base import DiscreteLoss


class ForwardLoss(DiscreteLoss):
    """Per-shot PDE-residual loss for the forward solve.

    L = w_pde * sum_s mean(r_pde_s ** 2), with `r_pde` returned by the
    `WaveEquation.residual`. Amplitudes are received already-spliced with
    the hard zero IC row (shape (n_shots, NT, NX, NY)).
    """

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor, shot_idx: int
    ) -> torch.Tensor:
        return self.config.wave_eq.residual(amp, wsp, self.sources[shot_idx])

    def evaluate(self, amp: torch.Tensor, wsp: torch.Tensor) -> torch.Tensor:
        n_shots = amp.shape[0]
        residuals = [self._residuals(amp[s], wsp, s) for s in range(n_shots)]

        w_pde = self.config.weights["pde"]
        L = w_pde * torch.stack([torch.mean(r**2) for r in residuals]).sum()

        self.evaluations += 1
        self._last_residuals = (torch.stack([r.detach() for r in residuals]),)

        return L
