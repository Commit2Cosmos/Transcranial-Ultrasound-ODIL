# No wavefield-block preconditioning, variable projection or WRI

The joint L-BFGS optimises the wavefield block and the model block together with no wavefield-block preconditioner, and `u` is never eliminated by variable projection or WRI. We chose this over three alternatives. The first was preconditioning `u` with `A(c₀)⁻¹` from cached Helmholtz factors (built in July, then removed with the block-coordinate optimiser in `7c130ad`). The other two, variable projection and WRI, eliminate `u`. That needs a new factorisation of `AᴴA + λPᵀP` at every model evaluation, which makes it reduced-space or extended-space FWI, a different inversion method rather than a preconditioner for this one. The `h⁻⁴` conditioning of the wavefield block is analysis only (#18 §3.2 item 4). The stall we have measured is in the model block while `u` absorbs the data misfit (#33), and a faster wavefield block would not fix that.

## Consequences

- The Helmholtz factor cache is removed (#31). A solver factorises `A(c)` only for the warm start of each band.
- A wavefield-block preconditioner is reconsidered only if, after #33 lands, an experiment on the same physical problem shows the L-BFGS iterations needed to reach a fixed relative c-error growing as `dx` shrinks, while the wavefield block's gradient dominates. It would keep the warm-start solve's factors for the band rather than restore the old cache (`abf1b01`, `554cad5`). Variable projection and WRI stay out of scope regardless.
