# Box bounds on the velocity in FWI, WRI and ODIL

Question: what methods exist for keeping the velocity inside a box `c_min ≤ c ≤ c_max` during inversion in reduced-space FWI, in WRI and its extended-source and ADMM variants, and in ODIL, and which of them carry over to this repo's joint ODIL solve? Scope: `SlownessLatent`, `JointFreqODIL._model_stats` and `init_params` (`odil_wave/optimisation/joint_odil.py`), `GridCfg.c_min` / `c_max` and `OptimiserCfg` (`odil_wave/experiment/config.py`), `HealthCheck` (`odil_wave/experiment/health.py`), ADR-0001 and ADR-0003, #35, #111, record 0002.

This file is about the **bound mechanism**. [model-reparametrisation.md](model-reparametrisation.md) (**MR**) already covers the choice of physical variable, the sigmoid's sensitivity and curvature tables (MR §3.1–3.3), and a first comparison of bound mechanisms against stock L-BFGS (MR §3.5). [joint-block-preconditioning.md](joint-block-preconditioning.md) (**JBP**) covers the curvature of the model block and why a variable change of variables doesn't fit L-BFGS (JBP §6). [near-source-artefacts-and-fringes.md](near-source-artefacts-and-fringes.md) (**NSA**) covers why the transducer cells collapse onto the lower bound. This file cites them and doesn't repeat them.

Domain terms follow `GLOSSARY.md`: **Model block**, **Wavefield block**, **Optimised variable**, **Model preconditioner**, **Wavefield preconditioner**, **Initial model**, **Warm start**, **True model**, **Inverse crime**, **Transducer**, **Interior**, **PML**, **Skull region**, **Brain region**, **PDE term**, **Data term**, **Regularisation term**, **Baseline**, **Protocol**. "Latent" means the optimised variable of the model block (`z_m` today). "Active" means a cell sits on a bound and the gradient pushes it outwards.

Every claim carries one provenance tag:

- **[S]**: established by a source. The key points to Sources. "(abstract)" means I reached only the abstract or a search excerpt. "(via X)" means I read the claim only as X reports it.
- **[R]**: measured in this repo, in an issue or an experiment record, or read from its code.
- **[I]**: my own inference or derivation. Nothing in the literature backs it for this regime unless a tag says so.

This session ran no pilots, so there are no [P] claims.

## 1. Conclusion

Five results decide the recommendations.

1. **In this repo the bounds are a safeguard, not a prior, and they bind only on the wrong path.** The true model spans 1450–3000 m/s inside the 1300–3100 m/s box [R: `velocity_models.py`, `config.py`]. So no bound is active at the solution. Record 0002's Baseline still hit `c_min` from iteration 5 and `c_max` from 55, while the wavefield preconditioner arm had `sat_frac` 0 and `c` in [1397, 2583] m/s [R: record 0002]. In #111 every bound hit is a cell whose truth is 1500 m/s [R: #111]. The literature uses bounds in this role too: Lucka et al. assume theirs "are not active at the vicinity of the solution" and "merely safeguard" the wave solver [S: Lucka22 App. D]. **Bound hits here are a symptom to fix upstream** (NSA §6, JBP), and the bound method matters mainly for what happens to a cell after it has hit [I].
2. **The sigmoid makes a hit hard to undo; a projection doesn't.** In the sigmoid latent the gradient of a cell at `sat_frac`'s threshold is scaled by `σ' ≈ 1e-3`, against 0.25 at mid-range. Its first-order escape is about 250× slower, whatever the sign of `∇_kΦ` (MR §3.1) [I]. Under a projection or L-BFGS-B, a cell on a bound leaves it as soon as the gradient points inwards: the bound is active only while the KKT sign condition holds [S: Byrd95 (abstract); Esser18 §II] [I]. Both are exact barriers to leaving the box. Only the projection lets #111's collapsed cells recover once the wavefield block catches up [I].
3. **Exact bounds and stock `torch.optim.LBFGS` don't mix.** Projecting an unconstrained quasi-Newton step "will not converge to a solution … for general `Hⁿ`" [S: Esser18 §II, via Bertsekas]. The working fixes all change the optimiser: a Cauchy point plus subspace step (L-BFGS-B) [S: Byrd95 (abstract)], a constrained quadratic model solved by SPG (PQN) [S: Schmidt09 §1–2], or a diagonal scaling, where the projected step has a closed form [S: Esser18 §II]. Torch's L-BFGS has no bounds argument [S: torchLBFGS docs]. ODIL calls SciPy's L-BFGS-B but passes **no bounds**, and bounds its parameters by a sigmoid [S: ODILgh `optimizer.py`, `heat.py`]. So the method closest to this repo, ODIL itself, does what the repo does today.
4. **WRI enforces bounds outside the wavefield update.** Aghamiry et al. split the model `m` from an auxiliary `p₀` and project only `p₀` with `min(max(·, m_l), m_u)`. The model update stays an unconstrained linear least-squares solve, coupled to `p₀` by an augmented-Lagrangian term [S: Aghamiry19 §2.2.2, eq. 26–29]. Bounds alone left periodic noise on the BP salt; bounds plus TV removed it [S: Aghamiry19 §3.3–3.4]. Peters & Herrmann find the same for FWI and WRI: bounds alone are "insufficient", bounds plus TV sharpen the salt [S: PetersHerrmann17]. **No source fixes cycle skipping or artefacts with bounds alone** [I, from the above].
5. **The optimiser decides which bound method is cheap.** Adam is "invariant to diagonal rescaling of the gradients" [S: Kingma15 (abstract)], so it largely cancels the sigmoid's `σ'` in the step on the latent, though not in `c` [I]. For a box the weighted projection used in Adam's convergence analysis [S: Reddi18 §2] reduces to a plain clip, because the box and the metric are both separable [I]. L-BFGS needs one of result 3's fixes. A Newton solve with exact per-cell curvature (as in ODIL's Newton option, JBP §6) would scale the latent step by `1/σ'`, undoing the trap. Its bounded form is the primal-dual active-set method, which is mesh-independent in 1D FWI [S: BurGh09] [I].

### Recommendations

| # | What | Why | Evidence | Cost |
|---|---|---|---|---|
| 1 | **Fix the cause before the bound.** Treat `sat_frac > 0` and `c` within 1% of a bound as a health failure, which `HealthCheck` already flags [R]. Rank NSA's transducer mask and the wavefield preconditioner (record 0002) ahead of any bound change. | Result 1: hits come from the path, not the solution. Record 0002's preconditioner arm cleared them with the sigmoid unchanged. | R; S: Lucka22 for bounds as safeguard | none |
| 2 | **Per-cell bounds for known regions** (water outside the head, the coupling medium at each transducer): a tight box, or `b = B`, which is a mask. `SlownessLatent` takes `m_min`/`m_max` as floats; tensors on interior cells broadcast with no other change [R: code]. A `b = B` cell must leave the latent, because the sigmoid's `Δm = 0` there. | Esser et al. and JUDI fix the water layer by equal bounds [S: Esser18 §III; JUDI scripts]. It hands NSA's transducer-mask option a bound-based form. | S (reduced space, seismic); joint space untested | S |
| 3 | **Raw-`k` latent with a soft C¹ box** (MR's candidate 1), clip `c` only for reporting and at band handover. | Result 2 without leaving stock L-BFGS. A frequency-domain band has no CFL, so small violations are harmless inside a band (MR §2.2 item 2). Peters & Herrmann's objection to penalties is about penalties that are active everywhere, not a box term that is zero inside [S: PetersHerrmann17] [I]. | MR; no joint-space ablation | S |
| 4 | **One diagnostic run with exact bounds**: SciPy L-BFGS-B on the stacked vector, through a wrapper like ODIL's [S: ODILgh `optimizer.py`], at the same `history_size`. Compare against candidate 3 on `rmse_brain`, `rmse_skull` and the count of cells on a bound. | Tells whether candidate 3's soft box loses anything against exact bounds. Float64 host copies make it slow (MR §3.5), so it is a check, not a default. | S: Byrd95, ODIL | M |
| 5 | **Revisit the box width once the skull moves.** `c_max = 3100` is 3% above the skull's 3000 m/s, where the sigmoid's sensitivity is 94 m/s per latent unit and 7.2 at `sat_frac`'s edge (MR §3.1). Under a projection or soft box the width doesn't change convergence, only the safeguard [I]. With the sigmoid, a wider `c_max` (real skulls reach 2800 cortical and about 3500 m/s in mandible and vertebrae [S: Guasch20]) moves the skull off the tail. | A single-key experiment, as MR §7 suggests. | I; S for tissue values | S |

**Not recommended now** [I]:

- **Clip after each `torch.optim.LBFGS` step** (Stride, Devito and Lucka's pattern, all with gradient descent or with bounds assumed inactive [S: Stride `ProcessModelIteration`; Devito `update_with_box`; Lucka22 App. D]). In this repo the bounds are active along the path (result 1), which is exactly where Esser et al.'s non-convergence warning applies. The strong-Wolfe line search would also run on unprojected trial points (MR §3.5).
- **Log barrier with L-BFGS:** trial points outside the box give `log` of a negative number; it needs a fraction-to-boundary step rule that torch's line search lacks. In Newton form, good preconditioners for interior-point systems "remain elusive" for FWI [S: BurGh09].
- **ADMM splitting of the model block** (result 4): it adds an alternating `p₀` update and multiplier, which `JointFreqODIL`'s design rules out ("no alternating updates, no … augmented-Lagrangian c-block") [R: module docstring]. It is the only joint-space precedent, though, so it is the form to use if bounds plus TV are ever needed (§5).
- **Multiple constraint sets (Dykstra, PARSDMM)** until a TV *constraint* is wanted in place of #111's TV penalty (§4.3).

## 2. Where the code stands

- **Map.** `m(z) = m_min + (m_max − m_min)·σ(z_scale·z)`, `m_min = 1/c_max²`, `m_max = 1/c_min²`, `c = m^{-1/2}` [R: `SlownessLatent`]. The module docstring says bounds are "enforced *structurally* by the sigmoid — every L-BFGS line-search trial stays in-bounds (no post-step projection)" [R].
- **Bounds.** `grid.c_min = 1300`, `grid.c_max = 3100` m/s. They also set the CFL time step through `c_max` [R: `GridCfg`]. `c_max` is the default `c₀` of the PDE's `k = (c/c₀)²` (MR §2.1) [R].
- **Initialisation and handover.** `init_params` maps the band's initial model to `z` with a logit clipped at `logit_clip = 1e-6` [R]. So a cell carried from the last band at `σ < 1e-6` restarts at `|z| ≤ 13.8`, still deep in the tail: `σ' ≈ 1e-6` [I].
- **Diagnostics.** `sat_frac` is the fraction of interior cells with `σ < 1e-3` or `σ > 1 − 1e-3` [R: `_model_stats`]. `HealthCheck` warns when `c` comes within 1% of a bound or `sat_frac` exceeds 1% (`BOUND_MARGIN`, `SAT_FRAC_LIMIT`) [R: `health.py`].
- **Optimiser.** One `torch.optim.LBFGS` over `(v_re, v_im, z_m)`, strong-Wolfe, `history_size = 20`, `inner_max_iter = 20` [R: `config.py`]. Torch's L-BFGS takes no bounds and supports a single parameter group [S: torchLBFGS docs].
- **`pde_diagonal`.** The model preconditioner uses `dk/dz` at the band's warm start [R: `model_curvature`], so it can't follow `σ'` as a cell saturates (JBP §5 "stale scale") [I].

## 3. Projection methods

### 3.1 Projection after an unconstrained step

- **How it works.** Take the optimiser's step, then clip each cell to `[b_i, B_i]`. For a box the Euclidean projection is the elementwise `min(max(·, b), B)` [S: Aghamiry19 eq. 29], written as an elementwise median of `(b, x, B)` in JUDI [S: JUDI `fwi_example_2D.jl` L42].
- **In code.** Stride's default `ProcessModelIteration` is a single `clip` step after each update [S: Stride `default_pipelines.py` L176–188, `steps/clip.py`]. Its breast examples clip to 1400–1700 m/s under `GradientDescent` [S: Stride `breast2D/02_script_inverse.py` L59–63]. Devito's FWI example updates `vp` with `Max(Min(vp + α·dm, vmax), vmin)` [S: Devito `inversion_utils.py` L20–31]. Lucka et al. project once at the end of each stochastic L-BFGS iteration, assuming the bounds are inactive near the solution [S: Lucka22 App. D, §4.1 (1350–1800 m/s)].
- **Failure mode.** With a non-diagonal scaling, projecting the unconstrained Newton or quasi-Newton step doesn't converge in general [S: Esser18 §II, via Bertsekas]. L-BFGS-B and orthant-wise descent "crucially rely on the separability of the constraints", which is what keeps their scaled direction a descent direction after projection [S: Schmidt09 §1]. Bertsekas's projected Newton makes the scaling diagonal on the binding set, so many constraints can enter and leave each iteration [S: Bertsekas82 (abstract)].
- **Interaction with the joint solve [I].** The clip touches only the model block, but the L-BFGS secant pair `(s, y)` then mixes the projected `z` with an unprojected `u` step. With gradient descent (Stride, Devito) there is no history to corrupt. Under strong Wolfe the line search sees the unprojected trial points.
- **Transfer [I].** Safe only when the bounds don't bind on the path, which result 1 says isn't the case here.

### 3.2 Scaled gradient projection with a diagonal metric

- **How it works.** Solve `min_Δ ∇fᵀΔ + ½Δᵀ(H + cI)Δ` subject to `m + Δ ∈ C`. For diagonal positive `H` and a box this has the closed form `Δ_i = max(b_i − m_i, min(B_i − m_i, −Δ_i^g))`, with `Δ^g = (H + cI)⁻¹∇f` [S: Esser18 §II eq. 10]. Esser et al. pick `c` adaptively, rejecting steps without sufficient decrease, as an implicit trust region that avoids line searches [S: Esser18 §II, Alg. 1]. Any limit point is stationary for the constrained problem [S: Esser18 §II].
- **Bounds used.** Squared slowness, spatially varying: the water layer is held fixed by `b = B`; elsewhere 1500–5500 m/s [S: Esser18 §III].
- **Transfer [I].** This is the only exactly bounded method with no history. The model block's Gauss-Newton diagonal is already computed for `pde_diagonal` [R]. But a diagonal-only step on the model block, with the wavefield block on L-BFGS, would be block-coordinate, which the solver's design excludes [R: module docstring]. Useful as a reference for what a bounded `pde_diagonal` would look like.

### 3.3 Spectral projected gradient (SPG)

- **How it works.** Projected gradient with Barzilai-Borwein spectral steps and a nonmonotone (Grippo-Lampariello-Lucidi) line search on convex sets [S: BMR00 (abstract); BMR14 (abstract)]. One projection per iteration onto the intersection of constraint sets [S: Peters18].
- **In code.** JUDI's 2D FWI examples run SPG from minConf with the box projection on squared slowness, 1.3–6.5 km/s, the water column fixed by equal bounds [S: JUDI `fwi_example_minConf.jl` L13–59], or with a PARSDMM projector onto bounds plus other sets [S: JUDI `fwi_example_constraints.jl` L58–121].
- **Cost.** Peters et al. note projected-gradient methods "are therefore often slower than Newton, Gauss-Newton, or quasi-Newton algorithms for FWI without constraints" [S: Peters18].
- **Transfer [I].** A gradient method on 5.4M stacked reals, where #96 found L-BFGS already needs thousands of iterations on the wavefield block alone (ADR-0003). Not competitive here.

### 3.4 L-BFGS-B and projected quasi-Newton (PQN)

- **L-BFGS-B.** Gradient projection with a limited-memory BFGS matrix for "large nonlinear problems with simple bounds" [S: Byrd95 (abstract)]. SciPy wraps a C translation of L-BFGS-B v3.0; bounds are `(min, max)` pairs per variable, default `m = 10` corrections [S: SciPy `fmin_l_bfgs_b`; ZhuByrd97; MoralesNocedal11]. NLopt's `LD_LBFGS` takes `lower_bounds!`/`upper_bounds!`, which JUDI uses for FWI on squared slowness [S: JUDI `fwi_example_NLopt.jl` L59–60].
- **PQN.** Builds a quadratic model with L-BFGS and minimises it over the original constraints with SPG, so it needs many projections per iteration. It pays off "when the cost of the projection is much lower than evaluating the function" [S: Schmidt09 §1–2]. It handles general convex sets where L-BFGS-B handles only boxes [S: Schmidt09 §1].
- **In ODIL.** ODIL's `LbfgsbOptimizer` calls `scipy.optimize.fmin_l_bfgs_b` with `m = 50` by default and no `bounds` argument. It converts every array to float64 numpy at each evaluation [S: ODILgh `optimizer.py` L27–101]. Its heat example bounds the conductivity as `sigmoid(knet)·kmax` [S: ODILgh `heat.py` L32–33]. The body-shape case uses a sigmoid-bounded body fraction [S: Karnakov23, via MR §4.1].
- **Transfer [I].** L-BFGS-B is the most direct exact-bound route and has an ODIL-shaped wrapper to copy. Costs: the host round trip and float64 on the stacked vector (MR §3.5), losing the persistent torch history between outer steps, and SciPy's own line search in place of torch's strong Wolfe. Bounds on `z` only; the wavefield block gets `(None, None)`. Suited to a one-off check (recommendation 4). No bounded L-BFGS exists in torch core [S: torchLBFGS docs].

### 3.5 Multiple constraints: POCS, Dykstra, PARSDMM

- **POCS** finds a point in the intersection, not the projection onto it, and its result depends on the order of the sets. **Dykstra** returns the projection, independent of order [S: Peters18].
- **PARSDMM** (SetIntersectionProjection.jl) projects onto intersections of up to about 16 convex or non-convex sets, beats Dykstra when single sets are hard to project onto, and has a multilevel scheme [S: PetersHerrmann19 (abstract); SIP README].
- **Results.** On FWI, "battle-tested multiscale workflows … fail even if we impose bound constraints"; bounds plus monotonicity and slope constraints, then relaxed to bounds alone, succeeded [S: Peters18]. Bounds 1425–5000 m/s there [S: Peters18].
- **Transfer [I].** A box alone is a separable clip, so these tools only matter once a second set, such as a TV ball, is added (§4.3).

## 4. Barriers, penalties and active sets

### 4.1 Log barrier and interior point

- Burstedde & Ghattas note that interior-point methods "often have better scaling properties" than active-set methods for discretised bound constraints, "but good preconditioners for solving the resulting optimal systems remain elusive" [S: BurGh09, Introduction].
- No FWI, WRI or ODIL source that uses a log barrier on velocity was found (§8).
- **Transfer [I].** The barrier `−μΣ[log(k − k_min) + log(k_max − k)]` is elementwise and smooth inside, and adds a diagonal term to `H_zz` that grows near the bounds. That repels cells from the bounds before they hit, the opposite of the sigmoid. But it needs a feasible start, a step cap for every line-search trial, and a `μ` schedule. Torch's L-BFGS has no step cap (MR §3.5).

### 4.2 Primal-dual active set (semismooth Newton)

- Burstedde & Ghattas enforce bounds on the elastic modulus "to satisfy physical constraints and to ensure stability of the wave-propagation solver". They use a primal-dual active-set strategy (Hintermüller et al.): a pointwise dual variable decides the upper, lower and inactive sets, and the Newton-Krylov system is solved over the inactive set only. With grid continuation the convergence is mesh-independent and needs "only a few additional CG iterations" [S: BurGh09, "Primal-dual active set strategy"; Hintermüller03 via BurGh09].
- **Transfer [I].** This fits ODIL's Gauss-Newton or Newton option (JBP §6), not L-BFGS. Freezing active model cells is a row and column deletion in the Newton system, cheap for a direct or algebraic multigrid solve. It is the natural bounded form if the repo ever adopts a Newton solver.

### 4.3 Penalty on bound violations

- **How it works [I].** Add `w_b·mean(relu(b − k)² + relu(k − B)²)` to the objective. It is C¹, zero inside the box, and works with stock L-BFGS and Adam. Violations scale like `|∇Φ|/w_b` at a stationary point, so bounds hold only approximately.
- **Against penalties.** Peters & Herrmann argue that penalties "may add undesired contributions to the gradients (and Hessians)" and need trade-off weights; constraints need only a projection [S: PetersHerrmann17]. That argument targets priors such as TV, which act everywhere. A box penalty is inactive in the interior of the box, so the distortion is limited to cells that violate [I].
- **TV as a constraint.** The same authors set the TV budget `τ` from the starting model and relax it by 1.25× per cycle [S: PetersHerrmann17]. #111 found TV penalty weights of 0.04 and 0.42 both unusable [R: #111]. A TV ball whose radius starts at the initial model's TV (zero for the homogeneous initial model) and grows is a different handle on the same problem [I]. It needs a projection onto the TV ball, so it rules out stock L-BFGS. Noted, not ranked.
- **Transfer.** MR ranks this first, on raw `k` (MR §1, candidate 1). This file agrees, for the safeguard role in result 1 [I].

## 5. Bounds in WRI and ADMM-based IR-WRI

- **Formulation.** Aghamiry et al. minimise `‖m‖_TV` over `u` and `m ∈ C = {m_l ≤ m ≤ m_u}`, subject to the wave equation, with an augmented Lagrangian. They alternate a closed-form wavefield update with a model update [S: Aghamiry19 §2.2, eq. 9–16].
- **Bound mechanism.** The model subproblem is linearised through the bilinearity of the Helmholtz operator, then split with auxiliary `p₀` (box) and `p₁, p₂` (TV) by split Bregman. The `m` update is an unconstrained sparse least-squares solve (eq. 26–27). The box enters only through `p₀ = proj_C(m − q₀)` with `proj_C(·) = min(max(·, m_l), m_u)`, plus a multiplier update [S: Aghamiry19 §2.2.2, eq. 21–29].
- **Results.** Box anomaly at 5 km/s: "bound constraints alone are not sufficient to reconstruct the bottom part of the anomaly and cancel out the oscillating artifacts" [S: Aghamiry19 §3.2]. BP salt, central target: bounds are switched on after 21 iterations; bounds alone leave periodic horizontal noise, BTV removes it, and the bounds "fasten the convergence of TV-regularized IR-WRI after iteration 20" [S: Aghamiry19 §3.3]. Left target: 441 iterations with bounds, 340 with BTV, and high-frequency noise in and under the salt with bounds alone [S: Aghamiry19 §3.4].
- **Penalty WRI.** Peters & Herrmann run bounds-only and bounds-plus-TV for both FWI and WRI. Bounds alone do "better, but … still unsatisfactory" in WRI [S: PetersHerrmann17].
- **Transfer [I].** The split keeps the model update unconstrained and pushes the box into a separable clip. In this repo the analogue is a penalty `(γ/2)‖k − p₀ + q₀‖²` in the joint objective, a clip of `p₀`, and a multiplier step between L-BFGS outer steps. That is an augmented Lagrangian on the model block, which the solver's design rules out [R: module docstring], and each `p₀` update changes the objective under the L-BFGS history (JBP §6). ADR-0003 forbids eliminating `u`, which this split doesn't do; the design statement is the obstacle, not ADR-0003.

## 6. Reparametrisation maps

MR covers the sigmoid in depth (MR §3). Additions here:

- **tanh** is an affine sigmoid (MR §1). **Softplus** and **exp/log** bound one side only, so they keep positivity of `k` but not `c_max` [I]. **sin²** creates a stationary point at every bound crossing (MR §1).
- **Clamp inside the map.** Torch's `clamp` with scalar bounds passes the gradient only where `min < x < max`, strictly, so a cell exactly on a bound gets zero gradient [S: PyTorch `FunctionsManual.cpp` `clamp_backward`]. A map `k = clamp(θ, b, B)` therefore freezes any cell that reaches a bound, a worse trap than the sigmoid [I]. A straight-through variant (identity gradient, clipped value) plus an in-place clip of `θ` is projected gradient on `θ`, with §3.1's failure mode under L-BFGS [I].
- **Squared-variable analogue.** For the substitution `x = y²`, first-order points of the reformulated problem "may not correspond to first-order optimal points" of the original, because the multipliers can take the wrong sign [S: DingWright23 (abstract)]. The sigmoid has no exact stationary point at finite `z`, but it has asymptotic ones in both tails regardless of the sign of `∇_kΦ`, the same defect in the limit [I].
- **Escape depends on the optimiser [I].** A per-cell Newton step in `z` is `−g_k/(σ'H_kk)` to first order, so exact diagonal curvature undoes the `σ'` trap. L-BFGS with history 20 and a scalar `H0` can't recover per-cell curvature over 15625 cells. Adam's per-coordinate normalisation makes the `z` step roughly independent of `σ'` [S: Kingma15 (abstract)], but the step in `c` is still `σ'` times smaller. MR §3.2 notes the sigmoid's second-derivative term can change the sign of `H_zz` near the bounds, which limits the Newton argument there.

## 7. Choice of bounds

| Source | Setting | Bounds [m/s] | Mechanism |
|---|---|---|---|
| This repo | Shepp-Logan, truth 1450–3000 | 1300–3100 | sigmoid [R] |
| Lucka22 | 3D breast USCT | 1350–1800 | projection per L-BFGS iteration [S] |
| Stride examples | 2D/3D breast | 1400–1700 | clip per gradient-descent step [S] |
| Esser18 | BP salt | 1500–5500, water fixed | scaled gradient projection [S] |
| Peters18 | seismic | 1425–5000 | SPG + Dykstra [S] |
| JUDI examples | Overthrust | 1300–6500, water fixed | SPG, NLopt L-BFGS, PARSDMM [S] |
| Guasch20 | transcranial, in vivo | none stated | gradient descent [S] |

- **Tissue values.** Soft tissue about 1400 (fat) to 1600 m/s (muscle); cancellous bone about 2100 and cortical about 2800 m/s; mandible and vertebrae about 3500 m/s [S: Guasch20]. The transcranial benchmark uses water 1500, brain 1560, trabecular bone 2300 and cortical bone 2800 m/s [S: Aubry22 Table 1].
- **Readings [I].**
  - The repo's `c_min = 1300` sits 150 m/s under the lowest truth and 100 m/s under fat. Raising it to about 1400 would cap how far #111's collapsed cells can fall. It would not stop the collapse, and with the sigmoid it would make the brain features at 1450 m/s sit closer to the lower tail.
  - `c_max = 3100` leaves 3% headroom over the phantom's skull. For a real skull it would sit below the mandible's 3500 m/s. Under the sigmoid the headroom sets the skull's final sensitivity (MR §3.1); under a projection or soft box it doesn't matter until it binds.
  - `c_max` also sets `dt` and the default `c₀` [R: `GridCfg`, `grid.py`], so changing it changes more than the box. An experiment should change only the latent's bounds or note the coupling.
  - Known regions take spatial bounds (recommendation 2). Every seismic code above fixes the water layer; the analogue here is the coupling water around and at the transducers.

## 8. Open questions and unverified

**Gaps in the literature.**

- No source compares bound mechanisms (sigmoid, projection, L-BFGS-B, barrier, penalty) in any joint-space inversion. ODIL uses a sigmoid without comparison [S: ODILgh; Karnakov23 via MR].
- No transcranial FWI source states its velocity bounds [S: Guasch20].
- No FWI, WRI or ODIL source using a log barrier on velocity was found.
- No source tests bounds whose activity comes from an optimisation-path artefact, as in #111, rather than from the prior.

**Not retrieved in full; abstract or search excerpt only:**

- Byrd95 (abstract; the generalised Cauchy point and subspace step are standard but not read in the paper), ZhuByrd97 and MoralesNocedal11 (cited via SciPy docs).
- Bertsekas82 (abstract and search summary only; the full-text PDF is a scan I couldn't extract). Esser18's non-convergence statement cites Bertsekas's textbook, which I didn't read.
- BMR00 and BMR14 (abstracts). PetersHerrmann19 (abstract). DingWright23 (abstract). Kingma15 (abstract only; the bound on Adam's effective step is not cited).
- Hintermüller et al. 2003, only through BurGh09.
- PetersHerrmann17 read through its HTML page; the WebFetch summary called the box projection an "elementwise median", which is the clip.

**Read in full or in the cited sections:** Schmidt09 §1–2, Esser18 §I–III, Aghamiry19 §2–3, Peters18 (first 100k characters), BurGh09 Introduction and the active-set section, Lucka22 §2, §4.1 and App. D, Reddi18 §2, Aubry22 Table 1, Guasch20 (via a page summary), and the code files listed in Sources.

## Sources

Retrieved in this session. Code is cited at the commit read.

- [Byrd95] Byrd R.H., Lu P., Nocedal J., Zhu C. "A limited memory algorithm for bound constrained optimization." *SIAM J. Sci. Comput.* 16(5):1190–1208 (1995). Abstract. https://doi.org/10.1137/0916069
- [ZhuByrd97] Zhu C., Byrd R.H., Lu P., Nocedal J. "Algorithm 778: L-BFGS-B." *ACM TOMS* 23(4):550–560 (1997). Via SciPy docs.
- [MoralesNocedal11] Morales J.L., Nocedal J. "Remark on Algorithm 778." *ACM TOMS* 38(1) (2011). Via SciPy docs.
- [SciPy] SciPy `fmin_l_bfgs_b` and `minimize(method='L-BFGS-B')` documentation. Read. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
- [Schmidt09] Schmidt M., van den Berg E., Friedlander M.P., Murphy K. "Optimizing costly functions with simple constraints: a limited-memory projected quasi-Newton algorithm." AISTATS 2009, PMLR 5:456–463. Read §1–2. https://proceedings.mlr.press/v5/schmidt09a.html
- [Bertsekas82] Bertsekas D.P. "Projected Newton methods for optimization problems with simple constraints." *SIAM J. Control Optim.* 20(2):221–246 (1982). Search excerpt. https://doi.org/10.1137/0320018
- [BMR00] Birgin E.G., Martínez J.M., Raydan M. "Nonmonotone spectral projected gradient methods on convex sets." *SIAM J. Optim.* 10(4):1196–1211 (2000). Abstract. https://doi.org/10.1137/S1052623497330963
- [BMR14] Birgin E.G., Martínez J.M., Raydan M. "Spectral projected gradient methods: review and perspectives." *J. Stat. Softw.* 60(3) (2014). Abstract. https://www.jstatsoft.org/article/view/v060i03
- [Esser18] Esser E., Guasch L., van Leeuwen T., Aravkin A.Y., Herrmann F.J. "Total variation regularization strategies in full-waveform inversion." *SIAM J. Imaging Sci.* 11(1):376–406 (2018). Read §I–III. https://arxiv.org/abs/1608.06159
- [PetersHerrmann17] Peters B., Herrmann F.J. "Constraints versus penalties for edge-preserving full-waveform inversion." *The Leading Edge* 36(1):94–100 (2017). Read through the HTML page. https://slim.gatech.edu/Publications/Public/Journals/TheLeadingEdge/2016/peters2016cvp/peters2016cvp.html
- [Peters18] Peters B., Smithyman B.R., Herrmann F.J. "Projection methods and applications for seismic nonlinear inverse problems with multiple constraints." *Geophysics* 84:R251–R269 (2019), DOI 10.1190/geo2018-0192.1. Read (first 100k characters of the HTML). https://slim.gatech.edu/Publications/Public/Journals/Geophysics/2018/peters2018pmf/peters2018pmf.html
- [PetersHerrmann19] Peters B., Herrmann F.J. "Algorithms and software for projections onto intersections of convex and non-convex sets with applications to inverse problems." arXiv:1902.09699. Abstract. https://arxiv.org/abs/1902.09699
- [SIP] SetIntersectionProjection.jl README. Read. https://github.com/slimgroup/SetIntersectionProjection.jl
- [Aghamiry19] Aghamiry H.S., Gholami A., Operto S. "Implementing bound constraints and total-variation regularization in extended full-waveform inversion with the alternating direction method of multipliers: application to large contrast media." *Geophys. J. Int.* 218(2):855–872 (2019). Read §2–3. https://arxiv.org/abs/1902.02744
- [BurGh09] Burstedde C., Ghattas O. "Algorithmic strategies for full waveform inversion: 1D experiments." *Geophysics* 74(6):WCC37–WCC46 (2009). Read the Introduction and "Primal-dual active set strategy". https://ins.uni-bonn.de/media/public/publication-media/BursteddeGhattas09.pdf?pk=572
- [Lucka22] Lucka F., Pérez-Liva M., Treeby B.E., Cox B.T. "High resolution 3D ultrasonic breast imaging by time-domain full waveform inversion." *Inverse Problems* 38:025008 (2022). Read §2, §4.1, App. D. https://arxiv.org/abs/2102.00755
- [Guasch20] Guasch L., Calderón Agudo O., Tang M.-X., Nachev P., Warner M. "Full-waveform inversion imaging of the human brain." *npj Digit. Med.* 3:28 (2020). Read through a page summary. https://pmc.ncbi.nlm.nih.gov/articles/PMC7060331/
- [Aubry22] Aubry J.-F. et al. "Benchmark problems for transcranial ultrasound simulation: intercomparison of compressional wave models." *J. Acoust. Soc. Am.* 152(2):1003–1019 (2022). Read Table 1. https://arxiv.org/abs/2202.04552
- [Kingma15] Kingma D.P., Ba J. "Adam: a method for stochastic optimization." ICLR 2015. Abstract. https://arxiv.org/abs/1412.6980
- [Reddi18] Reddi S.J., Kale S., Kumar S. "On the convergence of Adam and beyond." ICLR 2018. Read §2. https://arxiv.org/abs/1904.09237
- [DingWright23] Ding L., Wright S.J. "On squared-variable formulations." arXiv:2310.01784. Abstract. https://arxiv.org/abs/2310.01784
- [Karnakov23] See MR Sources. https://arxiv.org/abs/2303.04679
- [ODILgh] cselab/odil at `a59670e`: `src/odil/optimizer.py` L27–101 (`LbfgsbOptimizer`), `examples/heat/heat.py` L32–33 (`transform_k`). Read. https://github.com/cselab/odil
- [Stride] trustimaging/stride at `9a682b0`: `stride/optimisation/pipelines/default_pipelines.py` L176–188, `pipelines/steps/clip.py`, `constraints/bound.py`, `stride_examples/examples/breast2D/02_script_inverse.py` L56–63. Read. https://github.com/trustimaging/stride
- [Devito] devitocodes/devito at `448e7da`: `examples/seismic/inversion/inversion_utils.py` L20–31 (`update_with_box`). Read. https://github.com/devitocodes/devito
- [JUDI] slimgroup/JUDI.jl at `655cc44`: `examples/scripts/fwi_example_2D.jl`, `fwi_example_minConf.jl`, `fwi_example_NLopt.jl`, `fwi_example_constraints.jl`. Read the bound lines. https://github.com/slimgroup/JUDI.jl
- [torchLBFGS docs] PyTorch 2.15 `torch.optim.LBFGS` documentation. Read. https://docs.pytorch.org/docs/2.15/generated/torch.optim.LBFGS.html
- [PyTorch clamp] `tools/autograd/derivatives.yaml` (`clamp`) and `torch/csrc/autograd/FunctionsManual.cpp` (`clamp_backward`), `main` on 2026-11-09. Read. https://github.com/pytorch/pytorch
- [MR], [JBP], [NSA]: [model-reparametrisation.md](model-reparametrisation.md), [joint-block-preconditioning.md](joint-block-preconditioning.md), [near-source-artefacts-and-fringes.md](near-source-artefacts-and-fringes.md).
