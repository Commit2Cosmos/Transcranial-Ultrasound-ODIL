# No variable projection or WRI

The joint L-BFGS optimises the wavefield block and the model block together, and `u` is never eliminated by variable projection or WRI. Both need a new factorisation of `AᴴA + λPᵀP` at every model evaluation, which makes them reduced-space or extended-space FWI: a different inversion method, not a change to this one. That rules them out whatever the wavefield block's convergence turns out to be.

Preconditioning the wavefield block is a separate question and stays open. It does not eliminate `u`, so it is a change to the joint solve rather than a different method, in the same way `pde_diagonal` is for the model block. None is implemented today. The `A(c₀)⁻¹` preconditioner from cached Helmholtz factors (built in July, removed with the block-coordinate optimiser in `7c130ad`) was dropped because the stall measured then was in the model block while `u` absorbed the data misfit (#33). #93 has since shown that the wavefield block on its own converges slowly: with the model held at the true model, the problem is linear in `u`, yet L-BFGS needs about 200 × 20 iterations to reach a 5e-4 wavefield error on the unit problem (#96).

## Consequences

- The Helmholtz factor cache stays removed (#31). A solver factorises `A(c)` only for the warm start of each band.
- A wavefield-block preconditioner is added only when #96, or an experiment that follows it, shows that it cuts the iterations needed to reach a fixed wavefield error with the model held at the true model, measured on the same physical problem at two resolutions. A factor-based one keeps the warm-start solve's factors for the band rather than restoring the old cache (`abf1b01`, `554cad5`).
- A diagonal preconditioner alone is not expected to remove the `h⁻⁴` conditioning (#18 §3.2 item 4). That spread is between smooth and oscillatory modes, not between nodes, and the diagonal of `AᴴA` is nearly uniform in the interior. A diagonal can still even out the PML and the scale differences between shots and frequencies.
