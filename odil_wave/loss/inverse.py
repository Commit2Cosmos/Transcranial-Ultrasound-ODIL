from typing import Tuple

import torch

from odil_wave.wavefield import Wavefield

from .base import DiscreteLoss, mean_abs_sq
from .utils import LossConfig, LossTape


class InverseLoss(DiscreteLoss):
    """Joint (u, c) inverse-problem loss in the frequency domain.

    L = pde_weight  * mean(|r_pde|²)   # over shots, frequencies, space
      + data_weight * mean(|r_data|²)  # over shots, frequencies, receivers
      + w_reg  * R(c_interior)

    Global means keep the loss / gradient scale invariant to ``N_f`` and
    ``N_s`` so fair comparisons do not conflate more data with a larger loss.

    ``c_interior`` is the field passed to the regulariser. ``LBFGSB`` passes
    the normalised parameter ``c_new = c / c_ref`` so Tikhonov is scale-stable;
    the PDE always uses physical ``c_full``.

    Observations: full complex wavefields ``(n_shots, nf, nx, ny)`` or
    complex receiver traces ``(n_shots, nf, n_receivers)``.
    """

    def __init__(
        self,
        observed_wavefield=None,
        *,
        config: LossConfig,
        observed_traces=None,
        callback: LossTape | None = None,
        normalize_data: str | None = None,
    ):
        super().__init__(config, callback)

        has_wf = observed_wavefield is not None
        has_tr = observed_traces is not None
        if has_wf == has_tr:
            raise ValueError(
                "Supply exactly one of observed_wavefield or observed_traces."
            )

        cdtype = self.config.wave_eq.wavefield.cdtype
        self._trace_mode = has_tr
        if has_tr:
            tr = torch.as_tensor(observed_traces, dtype=cdtype)
            if tr.ndim != 3:
                raise ValueError(
                    f"observed_traces must be (n_shots, nf, n_receivers), "
                    f"got {tuple(tr.shape)}"
                )
            self.d_obs = tr.to(device=self.config.device)
        else:
            obs = self._stack_observations(observed_wavefield)
            self.d_obs = torch.as_tensor(obs, dtype=cdtype, device=self.config.device)

        self.normalize_data = normalize_data

    @staticmethod
    def _stack_observations(observed_wavefield):
        """Coerce input to a (n_shots, nf, NX, NY) tensor."""
        if isinstance(observed_wavefield, (list, tuple)):
            amps = [
                w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                for w in observed_wavefield
            ]
            return torch.stack(amps)
        return observed_wavefield

    def _obs_traces(self) -> torch.Tensor:
        """Observation traces as ``(n_shots, nf, n_receivers)``."""
        if self._trace_mode:
            return self.d_obs
        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        return self.d_obs[:, :, i, j]

    def _residuals(
        self, amp: torch.Tensor, wsp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pde = self.config.wave_eq.residual(amp, wsp, self.sources)

        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        syn_tr = amp[:, :, i, j]
        obs_tr = self._obs_traces()
        data = syn_tr - obs_tr
        return pde, data

    def evaluate(
        self,
        amp: torch.Tensor,
        c_full: torch.Tensor,
        c_interior: torch.Tensor | None = None,
        weights_override: dict | None = None,
    ) -> torch.Tensor:
        r_pde, r_data = self._residuals(amp, c_full)

        w = self.config.weights
        if weights_override is not None:
            w = {**w, **weights_override}
        # Global mean (not sum-over-shots): scale-invariant in N_f and N_s.
        pde_terms = mean_abs_sq(r_pde)
        data_terms = mean_abs_sq(r_data)
        L = w["pde"] * pde_terms + w["data"] * data_terms

        if self.config.regulariser is not None and c_interior is not None:
            L = L + w.get("reg", 0.0) * self.config.regulariser(c_interior)

        if L.is_complex():
            L = L.real

        self.evaluations += 1
        with torch.no_grad():
            pde_rms = float(mean_abs_sq(r_pde).sqrt().detach().cpu())
            data_rms = float(mean_abs_sq(r_data).sqrt().detach().cpu())
            src_rms = float(mean_abs_sq(self.sources).sqrt().detach().cpu())

        self._last_residuals = {
            "pde_rms": pde_rms,
            "data_rms": data_rms,
            "pde_loss": float(pde_terms.detach().cpu()),
            "data_loss": float(data_terms.detach().cpu()),
            "pde_src_ratio": pde_rms / max(src_rms, 1e-30),
        }

        return L

    def evaluate_z(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        c_full: torch.Tensor,
        c_interior: torch.Tensor | None = None,
        weights_override: dict | None = None,
    ) -> torch.Tensor:
        """Loss for ``u_precond='z'``: PDE in ``z``, data through ``u = A(c)^{-1} z``.

        Uses the same ``mean_abs_sq`` reduction as :meth:`evaluate` (no ``1/2``,
        no sum). Continuum ``½‖·‖²`` notation is conceptual only.

        * PDE residual: ``z - f'`` (``f'`` = :attr:`sources`)
        * Data residual: receiver sample of ``u`` minus observations
          (``u`` must be ``A(c)^{-1} z`` for the current ``c``)
        """
        r_pde = z - self.sources

        i = self.config.geometry.recv_ij[:, 0]
        j = self.config.geometry.recv_ij[:, 1]
        syn_tr = u[:, :, i, j]
        obs_tr = self._obs_traces()
        r_data = syn_tr - obs_tr

        w = self.config.weights
        if weights_override is not None:
            w = {**w, **weights_override}
        pde_terms = mean_abs_sq(r_pde)
        data_terms = mean_abs_sq(r_data)
        L = w["pde"] * pde_terms + w["data"] * data_terms

        if self.config.regulariser is not None and c_interior is not None:
            L = L + w.get("reg", 0.0) * self.config.regulariser(c_interior)

        if L.is_complex():
            L = L.real

        self.evaluations += 1
        with torch.no_grad():
            pde_rms = float(mean_abs_sq(r_pde).sqrt().detach().cpu())
            data_rms = float(mean_abs_sq(r_data).sqrt().detach().cpu())
            src_rms = float(mean_abs_sq(self.sources).sqrt().detach().cpu())

        self._last_residuals = {
            "pde_rms": pde_rms,
            "data_rms": data_rms,
            "pde_loss": float(pde_terms.detach().cpu()),
            "data_loss": float(data_terms.detach().cpu()),
            "pde_src_ratio": pde_rms / max(src_rms, 1e-30),
        }

        return L
