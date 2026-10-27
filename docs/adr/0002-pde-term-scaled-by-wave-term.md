# The PDE term is scaled by the wave term, not by the source

The PDE residual of each (shot, frequency) is divided by `ω'²·u_ref`, the RMS size of the wave term at the band's warm start (interior only, fixed for the band), instead of the RMS of the source `q`. We chose this because an RMS of `q` is only resolution-independent for a source whose width is fixed in metres: for a source resolved at grid scale (a point source, or a Gaussian tied to `h`) the RMS of `q` over the interior scales like `1/h` while a model-error residual stays O(1), so the PDE term drifts like `h²` against the data term, i.e. halving `dx` acts like a 4x data weight. The wave-term scale does not depend on the source's shape, amplitude convention or `nt`.

## Consequences

- The scale needs a non-zero warm-start wavefield, so every band is warm-started by a Helmholtz solve; a solver-free cold start (`u = 0`) is not supported.
- The PDE residual is summed over interior and PML nodes and normalised by the interior node count, so the PML counts in proportion to its physical size and grid size does not enter the balance.
