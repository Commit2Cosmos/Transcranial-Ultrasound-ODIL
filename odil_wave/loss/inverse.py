import torch

from odil_wave.wavefield import Wavefield

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


class InverseLoss:
    """Per-band inverse-problem data: operator, sources and observations.

    Holds what :class:`~odil_wave.optimisation.JointFreqODIL` reads to build
    its objective: the config's wave equation, geometry and regulariser, the
    ``t0**2``-scaled per-shot ``sources``, the observations
    (:meth:`obs_traces`) and the diagnostics ``callback`` tape.

    Observations: full complex wavefields ``(n_shots, nf, nx, ny)`` or
    complex receiver traces ``(n_shots, nf, n_transducers - 1)``; a shot is
    never recorded at its own transducer (``AcquisitionGeometry.receivers``).
    """

    def __init__(
        self,
        observed_wavefield=None,
        *,
        config: LossConfig,
        observed_traces=None,
        callback: LossTape | None = None,
    ):
        """Initialise the inverse loss from wavefield or trace observations.

        Parameters
        ----------
        observed_wavefield : optional
            Full complex wavefields ``(n_shots, nf, nx, ny)``; mutually
            exclusive with ``observed_traces``.
        config : LossConfig
            Loss configuration (keyword-only).
        observed_traces : optional
            Complex receiver traces ``(n_shots, nf, n_transducers - 1)``.
        callback : LossTape, optional
            Diagnostics tape.

        Notes
        -----
        Raises ``ValueError`` unless exactly one of ``observed_wavefield`` or
        ``observed_traces`` is supplied, or if traces are not 3-D.
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
                    for i in range(self.config.geometry.n_shots)
                ]
            )
            * t0**2
        )
        self.src_rms = float(mean_abs_sq(self.sources).sqrt())

        has_wf = observed_wavefield is not None
        has_tr = observed_traces is not None
        if has_wf == has_tr:
            raise ValueError(
                "Supply exactly one of observed_wavefield or observed_traces."
            )

        self._trace_mode = has_tr
        if has_tr:
            tr = torch.as_tensor(observed_traces, dtype=cdtype)
            if tr.ndim != 3:
                raise ValueError(
                    f"observed_traces must be (n_shots, nf, n_transducers - 1), "
                    f"got {tuple(tr.shape)}"
                )
            self.d_obs = tr.to(device=self.config.device)
        else:
            obs = self._stack_observations(observed_wavefield)
            self.d_obs = torch.as_tensor(obs, dtype=cdtype, device=self.config.device)

    @staticmethod
    def _stack_observations(observed_wavefield):
        """Coerce observations to a ``(n_shots, nf, NX, NY)`` tensor.

        Parameters
        ----------
        observed_wavefield : Wavefield, tensor, or sequence thereof
            Observed wavefield(s) in any accepted form.

        Returns
        -------
        torch.Tensor
            Stacked wavefield amplitudes.
        """
        if isinstance(observed_wavefield, (list, tuple)):
            amps = [
                w.amplitude if isinstance(w, Wavefield) else torch.as_tensor(w)
                for w in observed_wavefield
            ]
            return torch.stack(amps)
        return observed_wavefield

    def obs_traces(self) -> torch.Tensor:
        """Observation traces sampled at the receivers.

        Returns
        -------
        torch.Tensor
            Complex traces ``(n_shots, nf, n_transducers - 1)``.
        """
        if self._trace_mode:
            return self.d_obs
        return self.config.geometry.extract_observations(self.d_obs)
