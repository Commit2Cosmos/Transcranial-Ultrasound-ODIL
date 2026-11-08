# Reparametrising the model block of the joint ODIL solve

Question: which reparametrisation of the optimised model variable would most improve convergence and reconstruction in this repo's frequency-domain joint ODIL inversion, and what does the literature show in similar problems? Scope: `SlownessLatent` and `JointFreqODIL` (`odil_wave/optimisation/joint_odil.py`), `WaveEquation.residual` (`odil_wave/operator/utils.py`), ADR-0001 to ADR-0003, #18 §3.4, #35, #33, #97, #98, #111, record 0002.

This file extends [joint-block-preconditioning.md](joint-block-preconditioning.md) §5, "Change of parameterisation" (cited below as **JBP §n**). It doesn't repeat that file's block-Hessian derivation (JBP §2) or its preconditioner survey (JBP §3–§7).

Domain terms follow `GLOSSARY.md`: **Model block**, **Wavefield block**, **Model preconditioner**, **Wavefield preconditioner**, **Initial model**, **Warm start**, **True model**, **Interior**, **PML**, **Skull region**, **Brain region**, **PDE term**, **Data term**, **Regularisation term**, **Baseline**, **Protocol**, **Experiment spec**. "Latent" means the variable L-BFGS optimises for the model block (`z_m` today). "Map" means the function from latent to the physical model.

Every claim carries one provenance tag:

- **[S]**: established by a source. The key points to Sources. "(abstract)" means I reached only the abstract or a search excerpt.
- **[R]**: measured in this repo, in an issue or an experiment record.
- **[P]**: measured in a scratch pilot in this session: float64 linear algebra on uncommitted code, with no L-BFGS. It indicates, it does not establish. Under the glossary's rule for scratch runs, no experiment record may cite it.
- **[I]**: my own inference or derivation. Nothing in the literature backs it for this regime unless a tag says so.

This session ran no pilots, so there are no [P] claims. Numbers tagged [I] are closed-form arithmetic at the default bounds (1300–3100 m/s), water 1500 m/s, skull 3000 m/s.

## 1. Conclusion and ranked recommendation

Four results decide the ranking.

1. **The brief's premise is inverted for this code.** The discrete residual is `r = λ_tt u − k ∇'²u − f'` (plus PML terms) with `k = (c/c₀)²` (`WaveEquation.residual`; `c₀ = c_max` by default). So `r` is **bilinear in `(k, u)`**, not in squared slowness [R: code, #18 §3.4]. In `m` it is `λ_tt u − ∇'²u/(c₀²m) − f'`, which is nonlinear in `m` [I]. Squared slowness is the bilinear variable only in the `ω²m u + ∇²u` form used by WRI and IR-WRI [S: vLH13 eq. 1; Aghamiry19 §2.1]. The two forms differ by the row scaling `diag(k)`, which changes the PDE term, not the model variable (§2.1).
2. **Under the sigmoid, the choice of physical variable is mostly irrelevant.** Squared slowness, `k`, and any contrast affine in either are related by Möbius maps. Sigmoid bounds on any of them give the same problem up to an affine change of latent, and L-BFGS with scalar `H0` follows the same trajectory [R: #18 §3.4, verified to 9.1e-13 m/s; I for the generalisation to contrasts]. **What matters is the bound map, and whether the map is pointwise or spatial.**
3. **No pointwise map can touch JBP result 1.** A pointwise map applies the same diagonal congruence to `H_zz` and `S`, so `μ = eig(H_zz⁻¹ᐟ²SH_zz⁻¹ᐟ²)`, `rank(S)` and `null(S)` stay unchanged at any given point [I]. Pointwise maps change three things only: the nonlinearity along the path, how stale a fixed warm-start scale gets, and spurious stationary points at the bounds (§3). Only **spatial** maps (multilevel, splines, networks) change `S` itself. By JBP §2, about half of model space at one frequency has no Gauss-Newton curvature, so spatial maps are the only reparametrisations that can act on it [I].
4. **The sigmoid traps cells at the lower bound, not the skull.** The skull at 3000 m/s keeps `|dc/dz| = 94` m/s per latent unit, against 173 for water and a peak of 431 near 2200 m/s [I]. A cell at the `sat_frac` threshold has 0.54 m/s at the 1300 m/s bound, and 7.2 at 3093 m/s [I]. Record 0002's Baseline put cells on the 1300 m/s bound from iteration 5, and on the 3100 m/s bound from 55 [R]. After 100 outer iterations the skull's mean was still only 1866 m/s, where the sigmoid is near its steepest [R: #111]. So, today, saturation matters through the trapped cells in the water and brain, and through the sigmoid's own curvature (§3.2). It doesn't slow the skull until the skull is nearly recovered.

**No source tests any model reparametrisation in a joint (full-space, penalty or ODIL) Helmholtz inversion at skull contrast.** Joint-space evidence covers only the variable choice (squared slowness, chosen for bilinearity [S: vLH13, vLH16, Aghamiry19]), box projection with TV on salt [S: Aghamiry19], ODIL's sigmoid-bounded field [S: Karnakov23], and multilevel decomposition in ODIL [S: Karnakov23]. Everything else comes from reduced-space FWI, breast USCT or PINNs. §4 flags each item.

### Ranked plan

| # | Candidate | Why this rank | Evidence | Cost |
|---|---|---|---|---|
| 1 | **Raw `k` latent with a soft box** (#35), Jacobian-matched at the initial model | Exact bilinearity, no map curvature, no trap at the bounds, `H_zz` exact away from `r = 0`. Elementwise, so it fits stock L-BFGS and the GPU. | Joint space: bilinear variable [S: vLH13, Aghamiry19]; no ablation anywhere | S |
| 2 | **Log-velocity latent with a soft box**, Jacobian-matched | The only power-law variable whose PDE curvature per cell doesn't depend on `c` at `r = 0` (§3.3). So `pde_diagonal` stays valid as the skull speeds up. Mild map curvature. | Reduced space: log (Jeffreys) parameters [S: secondary, Tarantola via SEP143]; the invariance is [I] | S |
| 3 | **Multilevel latent** (mODIL decomposition of the model block) | The only cheap spatial map. Overparameterised and linear, so it fits the stacked L-BFGS. Acts on `null(S)` and on #111's fringes. | ODIL, joint, L-BFGS-B [S: Karnakov23], but not ablated on a parameter field | M |
| 4 | **Coarse or B-spline model grid** (fixed, for example 4 mm knots) | Cuts null directions and the `h²` drift outright (JBP §2), but limits the 4 mm skull rim | Reduced space: coarse-to-fine [S: Lucka22, FWIME22 (abstract)] | M |
| 5 | **Projected or L-BFGS-B box on raw `k`** | Exact bounds, but not in torch: needs scipy on the CPU or a custom projected L-BFGS | Reduced space: projection [S: Lucka22, Esser18 (abstract)]; joint space via ADMM [S: Aghamiry19] | M–L |
| 6 | **Neural implicit representation** (SIREN, Fourier features) | Smoothness bias against a 4 mm rim; Adam-based, with no L-BFGS evidence on a stacked joint vector | Reduced space [S: IFWI22, P2INR25]; PINN [S: RashtBehesht22] | L |
| 7 | **Deep image prior** (CNN generator) | As 6, plus a generator architecture to choose; the salt result came out "smooth and approximate" | Reduced space [S: Zhu22, WuMcMechan19 (abstract)] | L |

Candidates 1 and 2 share one feature under test and can share one experiment: three values of one key, with the current latent as Baseline (§6).

### Ruled out, and why

- **Sigmoid on `k`, or on any contrast affine in `m` or `k`** (`m/m₀ − 1`, CSI's contrast, `k/k₀ − 1`): identical to today's latent up to an affine latent [R: #18 §3.4; I for contrasts].
- **tanh**: `tanh(x) = 2σ(2x) − 1`, so it is the sigmoid with an affine latent [I].
- **Raw `m`, raw slowness, raw velocity**: dominated. `m` and `s` lose bilinearity and also go stale in the wrong direction (§3.3). `c` has less map curvature than `log c`, but its curvature scale still drifts 4× across the skull [I].
- **Clipping inside the map** (`k = clamp(θ)`): zero gradient beyond the bound and a kink at it. Either can stall the strong-Wolfe line search [I].
- **Periodic map** (`k = k_min + Δk sin²θ`): `dk/dθ = 0` at every bound crossing, so it creates a spurious stationary point at each one. Extra, equivalent latents at each period [I]. No FWI source found.
- **Mirror descent or exponentiated gradient** on the box: equals a step in logit coordinates driven by `∇_kΦ`, not `∇_zΦ`, i.e. the sigmoid latent with a per-cell scale `1/σ'` refreshed every iteration [I]. A variable scale isn't a fixed change of variables, so stock L-BFGS can't use it (JBP §6).
- **Switching to the `m`-form residual**: this changes the PDE term, not the model variable. Out of scope (§2.1).

## 2. Where the code stands, and corrections to JBP §5

### 2.1 What is bilinear

- `WaveEquation.residual` computes `λ_tt·u − (c/c₀)²·∇'²u − source`. `assemble_H_sparse` builds `H = diag(a) − diag(c_nd²)L` to match [R: code]. `JointFreqODIL` builds `c` as `m^{-1/2}` (`c_of_z`) and the residual squares it again, so the PDE sees `k = 1/(c₀²m)` [R: code; #18 §3.4 item 3].
- WRI writes `A(m) = ω²diag(m) + L`, linear in squared slowness [S: vLH13 eq. 1]. The penalty-method paper notes that, for linear PDEs, `A` "can often be made to depend linearly on m" [S: vLH16]. IR-WRI defines its model as squared slowness, which makes the model subproblem linear "due to the bilinearity" [S: Aghamiry19 §1, §2.1]. PINN FWI uses the `α²∇²φ = φ_tt` form, which is linear in `α²` [S: RashtBehesht22, appendix].
- The two discrete forms are the same PDE, divided by `c²` [I]. Dividing the residual rows by `k` would make it bilinear in `m`. It would also reweight the PDE term per cell by `1/k²` (16× between water and skull) and rescale the source term, both against ADR-0002's design [I]. That is a different objective, and belongs to ADR-0002 and #80, not to reparametrisation.

### 2.2 What JBP §5 says, and what changes here

JBP §5, "Change of parameterisation": `r` is linear in `k`, the sigmoid-bounded `k` is an affine relabelling of the sigmoid-bounded `m` [R: #18], "so only a projected raw `k` (#35) changes anything", and a raw `k` makes any model preconditioner less stale. This file adds:

1. **The equivalence class is larger** (result 2): any variable Möbius-related to `k` under matching sigmoid bounds. Velocity, slowness and `log c` are not in it, so sigmoid bounds on those are different problems [I]. None is better than raw `k` or raw `log c` (§3).
2. **Projection isn't the only way to use raw `k`.** In the frequency domain there is no CFL condition inside a band. The residual can be evaluated for any real `k`, so bounds only need to hold where `c` must be physical: in reporting, in the regulariser, and at the next band's warm start [I]. A soft C¹ box penalty fits stock L-BFGS. Lucka et al. likewise use bounds "merely" to keep their time-stepping scheme stable, assuming they aren't active at the solution [S: Lucka22, App. D].
3. **Pointwise maps leave `μ` unchanged** (result 3). This bounds what any variable choice can do against JBP result 1 [I].
4. **The staleness of the warm-start scale has a closed form** in each variable (§3.3). JBP §5's "stale scale" bullet is the sigmoid instance.
5. **Unscaled, the variable choice changes the global block balance.** At the water initial model `dk/dz_m = 0.054` (in `c₀ = c_max` units), so a raw `k` latent has 341× the PDE curvature of `z_m`, and `log c` has 75× [I]. Against #33's `median(H_uu)/median(H_zz) ≈ 1.5e4` [R], that alone acts as a strong model preconditioner. To isolate the map from the scale, candidates 1 and 2 are **Jacobian-matched**: a fixed per-cell factor makes `dk/dθ` at the initial model equal to the sigmoid's (§6).

## 3. Pointwise maps: physical variable and bound map

### 3.1 Sensitivity of the current map

`|dc/dz_m| = (c³/2)·(m_max − m_min)·σ(1 − σ)`, with `z_scale = 1` [I, from `SlownessLatent`]:

| `c` [m/s] | 1300.5 (`sat_frac` edge) | 1450 (brain features) | 1500 (water, initial) | 2000 | 2200 (peak) | 3000 (skull) | 3093 (`sat_frac` edge) |
|---|---|---|---|---|---|---|---|
| `\|dc/dz_m\|` [m/s per unit] | 0.54 | 135 | 173 | 409 | 431 | 94 | 7.2 |

Readings [I]:

- **A cell pinned near 1300 m/s has 1/320 of the water sensitivity.** Its gradient vanishes whatever `∇_kΦ` is: `∇_zΦ = (dk/dz)∇_kΦ → 0` is a stationary point the map creates, not the physics. This is the "bounds hit at iteration 5" mechanism of record 0002's Baseline [R for the event, I for the mechanism].
- At 3000 m/s the sigmoid keeps 54% of the water sensitivity, so it would slow the skull only once the skull is within about 100 m/s of `c_max`.
- Real skulls reach 2800 m/s (cortical) and 3500 m/s (mandible, vertebrae) [S: Guasch20], so a narrower `c_max` would push skull cells into the 3093 m/s regime [I].

### 3.2 Nonlinearity from the map

With `k = k(θ)`, the model-block Hessian gains `a·Re(r̄·(−∇'²u)·k''(θ))` beyond its Gauss-Newton part `a(k')²|∇'²u|²` (JBP §2, discrepancy 2). Their ratio is the dimensionless `ρ = k·k''/(k')²` times the relative PDE residual [I]:

| Latent | `ρ` | Value at 1500 / 3000 / 3086 m/s |
|---|---|---|
| raw `k` | 0 | 0 / 0 / 0 |
| raw `c` | 1/2 | 0.5 |
| `log c` | 1 | 1 |
| raw slowness | 3/2 | 1.5 |
| raw `m` | 2 | 2 |
| sigmoid (today) | `k(1 − 2σ)/(Δk σ')` | 3.7 / −13.5 / −109 |

Readings [I]:

- Record 0002's Baseline ends with a `pde_rel_residual` of 9.8e-3 [R]. Near the upper bound the sigmoid's term is then about the size of the Gauss-Newton term, with either sign, so the model block can lose positive-definiteness there. JBP §2 found the Gauss-Newton form off by 49% at a perturbed point [P, in JBP].
- Raw `k` removes the term and keeps `H_zz` exact at any `r`. Positivity then needs the soft box: `k ≤ 0` would make `c` imaginary.
- `log c` keeps the term at O(1) everywhere and is positive by construction.

### 3.3 Staleness of the warm-start scale

Away from sources and the PML, at `r = 0`, `∇'²u = (λ_tt u − f')/k = λ_tt u/k`. So `H_kk ∝ |λ_tt u|²/k²` exactly, with no plane-wave assumption [I]. With `H_θθ = (dk/dθ)²H_kk`, at fixed `|u|`:

| Latent | `H_θθ ∝` | Skull ÷ water at the true model (×) |
|---|---|---|
| raw `k` | `1/k²` | 1/16 |
| raw `c` | `1/c²` | 1/4 |
| **`log c`** | **1** | **1** |
| raw slowness | `c²` | 4 |
| raw `m` | `c⁴` | 16 |
| sigmoid (today) | `(dk/dz/k)²` | 1/13.7 |

Readings [I]:

- `pde_diagonal` sets `s` at the warm start, where `k` is uniform (`_build_model_scale`) [R: code]. Once the skull reaches 3000 m/s, its curvature in today's latent is 13.7× lower than `s` assumes. In `log c`, only the change in `|u|` remains.
- This is the parameter-space version of Tarantola's argument that positive quantities such as velocity and slowness are best handled through their logarithms [S: Tarantola via SEP143 (secondary); TarantolaValette82 via the same]. That argument rests on invariance and is a statistical one; the curvature identity above is mine.
- None of this changes `μ` (result 3). It decides only whether a fixed diagonal model preconditioner stays right for the band.

### 3.4 Cycle skipping

- A pointwise diffeomorphism maps local minima of the objective to local minima. It can't remove a cycle-skipped basin; it changes only which basin a given path reaches [I]. Maps whose derivative vanishes (sigmoid tails, `sin`, clipping) add stationary points that the objective in `c` doesn't have [I].
- In joint space, robustness to cycle skipping comes from the extended search space, not from the variable [S: vLH13; Aghamiry19 abstract, and §2.1.1 on the penalty parameter]. Bound and TV constraints alone don't find the base of high-velocity salt from a poor initial model; an asymmetric TV constraint was needed [S: Esser18 (abstract), reduced-space and WRI].
- Reduced space, gradient descent: velocity and squared slowness give gradients differing by `−2/v³`, and so "distinct convergence properties". Pseudo-Hessian preconditioning makes the two updates agree for small steps [S: Carneiro14, Marmousi-2, steepest descent]. Reduced space, 3D breast USCT, stochastic L-BFGS: slowness and squared slowness "speed up the convergence" over sound speed, while linear spatial weightings made no significant difference, which the authors attribute to quasi-Newton's low sensitivity to linear transforms [S: Lucka22 §4.8]. That contrast range (1350–1800 m/s bounds [S: Lucka22 §4.1]) is far below a skull's. **Neither result transfers directly:** both are reduced-space, and neither is near 2:1 contrast [I].

### 3.5 Fit with L-BFGS, the GPU and the ADRs

| Bound mechanism | Smooth? | Stock `torch.optim.LBFGS`? | GPU | ADR notes |
|---|---|---|---|---|
| Sigmoid (today) | C^∞, traps at bounds | Yes | Elementwise | — |
| Soft box penalty `w_b·mean(relu(k_min−k)² + relu(k−k_max)²)` | C¹ | Yes | Elementwise | `w_b` a library default (ADR-0001); not a regularisation term in the glossary's sense |
| Log barrier | C^∞ inside the box | Line-search trials can leave the box, giving NaN, so it needs a step cap | Elementwise | Custom guard |
| Projection after each step | Kinked | No: corrupts the secant pairs, and the line search sees unprojected points | — | Custom projected L-BFGS |
| L-BFGS-B (scipy) | — | Replaces torch's L-BFGS for the whole stacked vector, 5.4M reals | CPU, float64, a host copy every evaluation | Loses persistent torch history |

[I] throughout, apart from ODIL itself having used scipy's L-BFGS-B [S: Karnakov23 §2.1] and torch's L-BFGS having no bounds [S: torchLBFGS, via JBP Sources].

- **ADR-0002:** `pde_scale = |λ_tt|·u_ref` doesn't depend on the model, so no pointwise map touches it. The regulariser acts on `c`, so raw `k` needs `c = c₀√max(k, ε)` there (`reg = 0` by default) [I, R: code].
- **ADR-0003:** none of these eliminates `u` [I].
- **ADR-0001:** a new `RunConfig` key (for example `optimiser.model_latent`) whose default keeps `SlownessLatent`. `z_scale`, `logit_clip` and `w_b` stay library defaults [I].
- **Per-evaluation cost:** a few elementwise ops on 15625 cells, against a 0.145 s evaluation [R: record 0002]. Negligible [I].

## 4. Spatial maps

All of these replace the per-cell latent with `θ = Φ(w)` for a fixed (multilevel, spline) or learned (network) `Φ`, composed with a pointwise map from §3 [I].

**Scale argument [I].** At 80 kHz, λ/2 in water is 9.4 mm (JBP §5), against 2 mm cells. `rank(S) ≤ 8064`, likely about 4000, against 15625 cells (JBP §2). A 63² coarse level has 3969 cells. So a level near 4 mm already matches the information one band provides. The skull rim is 4 mm thick, also under λ/2 at one band.

### 4.1 Multilevel latent (mODIL)

- **Source [S: Karnakov23 §2.2].** `u = u₁ + T₁u₂ + … + T₁⋯T_{L−1}u_L`, linear interpolation, overparameterised. The rest of ODIL, including the optimiser, is unchanged. It "extend[s] the domain of dependence of each scalar parameter so that information can propagate through the grid faster". With L-BFGS-B (history 50) on Poisson, five levels reach machine precision in 200 iterations in 2D, where Adam stalls. Conductivity inference: 1500 against 4000 L-BFGS-B iterations to 5% error, with monotone error, "therefore, the multigrid decomposition regularizes the solution". The abstract claims an order-of-magnitude speedup and better avoidance of local minima.
- **Flag: joint space, ODIL, L-BFGS-B, but the decomposition was applied to the *state* fields.** The conductivity was a small network of `u`, squared for positivity. The body-shape case applies it with six levels and a **sigmoid-bounded** body fraction, `χ = 1/(1 + e^{−(χ̂+5)})`, but with no ablation [S: Karnakov23 §3.4–3.5]. That last case is the nearest precedent for today's latent. **No ODIL source ablates a multilevel parameter field.**
- **Fit [I].** It is a fixed linear map, so L-BFGS history stays valid (JBP §6). Torch `interpolate` or `conv_transpose2d` is cheap on MPS and CUDA. Parameters grow 1.33×. `pde_diagonal`'s `s` would apply to the summed fine field: `z_m = s ⊙ M(y₁…y_L)`. With uniform level weights, coarse levels see about `4^l` times the curvature, which changes the block balance like a preconditioner; Karnakov et al. don't rescale them.
- **Expected effect [I].** Coarse components carry the low-wavenumber skull and background moves that L-BFGS with history 20 reaches slowly per cell. It should damp #111's λ/2 fringes in the water. It can't add curvature along `null(S)`, but L-BFGS's minimum-norm-like path prefers the coarse levels, which acts as implicit regularisation.

### 4.2 Fixed coarse grid or B-splines

- Reduced space: a coarse, possibly non-uniform spline grid mapped to the FD grid by a sparse linear operator, refined over the iterations [S: FWIME22 (abstract)]. 3D breast USCT, coarse-to-fine across four levels: the two-level scheme beat single-level's 16-hour error sooner, but each level transition smooths and can raise the error [S: Lucka22 §3.3, §4.7]. Lucka's coarsening is of the whole simulation, not only the model.
- Wavelet parameterisation: no primary source retrieved (§8).
- **Fit [I].** A linear map that removes null directions outright and stops the `h²` drift of `H_zz/diag(S)` (JBP §2). It costs resolution at the rim. Refining within a band is a variable map, so it means a restart (JBP §6); refining per band needs no new code beyond the map.

### 4.3 Neural implicit representations

- **IFWI** [S: IFWI22]: a 4×128 SIREN with `ω₀ = 30` and linear output normalised by the Marmousi mean and standard deviation, so **unbounded**. Trained with Adam over thousands of epochs. From a random initial model it is comparable to FWI from a smooth one, and it learns low wavenumbers first. Reduced space, time domain, Marmousi.
- **Spectral bias** [S: Tancik20]: an MLP's neural tangent kernel decays fast with frequency. Fourier features tune its bandwidth.
- **P2INR-FWI** [S: P2INR25, MICCAI page and reviews]: a polar SIREN with Fourier encoding for ring-array breast USCT, with a mask prior from reflections. Reviewers report it slower than conventional FWI (7.1 s against 5.6 s). It doesn't mention skull or high contrast.
- **PINN FWI** [S: RashtBehesht22]: a separate 5×20 velocity network beside the wavefield network, in joint space, unbounded. Velocity is scaled to [0, 1] by the PDE's normalisation. Adam, and "L-BFGS … did not [give] significant improvement".
- **Fit [I].** One evaluation of a 4×128 SIREN at 15625 points is tiny on the GPU. But no source runs L-BFGS on network weights stacked with millions of grid unknowns. The PINN result points against it, and the weight-space curvature differs from the wavefield block's by an unknown factor, which `pde_diagonal` can't set. The low-frequency bias works against a 4 mm rim at 2:1 contrast, unless high-frequency features are added, which brings back roughness.

### 4.4 Deep image prior

- **NNFWI** [S: Zhu22]: a CNN generator (tanh output, scaled) added to the initial model. In clean Marmousi, conventional FWI is better (MSE 114 against 152); with noise NNFWI is far better (171 against 568). On BP 2004 salt it beats hand-tuned TV (MSE 134 against 181), but the salt comes out as "a smooth and approximate shape". It reaches loss targets more slowly than L-BFGS FWI (240 against 570 iterations to 3%). Reduced space.
- Wu & McMechan reparametrise the initial model in CNN weights [S: WuMcMechan19 (abstract)]. A 2025 benchmark finds a pretrained CNN best among U-Net, CNN and MLP, and frames spectral bias as adaptive multiscale [S: DRFWI25 (abstract)].
- **Fit [I].** Same as 4.3, plus an architecture to choose. Its main reported benefit, noise robustness, isn't tested under the inverse crime.

### 4.5 Rigid or shape parameterisations (noted, not ranked)

Skull templates aligned through three SE(2) parameters, from FWI gradients by autodiff, with pixel FWI for the brain afterwards [S: MOFI26]. Shape derivatives on skull contours [S: Marty23 (abstract)]. Both need a template or a segmentation, which this repo's Shepp-Logan setting deliberately lacks [I]. In vivo transcranial FWI first builds the skull with AWI, then runs FWI, with no stated velocity bounds [S: Guasch20].

## 5. Joint against reduced-space evidence

| Claim | Joint space (penalty, WRI, ODIL, PINN) | Reduced space only |
|---|---|---|
| Bilinear variable eases the model step | vLH13, vLH16, Aghamiry19 | — |
| Box projection plus TV on high contrast | Aghamiry19 (BP salt, IR-WRI, ADMM) | Esser18 (salt), Lucka22 (bounds as safeguard) |
| Variable choice changes convergence | — | Carneiro14, Lucka22 |
| Log parameters | — | Tarantola (secondary) |
| Sigmoid-bounded field with L-BFGS | Karnakov23 (body fraction) | Zhu22 (tanh, Adam) |
| Multilevel decomposition | Karnakov23 (state fields; parameter field not ablated) | Lucka22, FWIME22 (sequential coarse-to-fine) |
| Neural representation | RashtBehesht22 (Adam) | IFWI22, P2INR25, Zhu22, WuMcMechan19 |
| Transcranial | — | Guasch20, MOFI26, Marty21/23 |

[I] for the classification, [S] per cell as cited.

## 6. Candidates for the `experiment` skill

Every candidate is a **feature under test** on an `exp/*` branch, behind one config key whose default keeps current behaviour (ADR-0001; PROTOCOL.md gate 2). Suggested key: `optimiser.model_latent`.

**Default Baseline:** record 0002's `none` arm, i.e. the default config (125² interior, PML 40, 64 transducers, homogeneous 1500 m/s initial model, one 80 kHz band, `model_preconditioner: none`, `wavefield_preconditioner: none`) with `optimiser.n_iter = 100` [R]. Its finals: `rmse_skull` 1152.0, `rmse_brain` 85.10, `bias_skull` −1133.7, `sat_frac` 0.0041, `rel_wavefield_error` 0.4514 [R].

**Jacobian matching** (candidates 1, 2) [I]. `k = k₀ + g₀·θ` (raw `k`) or `c = c_init·exp(g₀′·θ)` (`log c`). Per cell, `g₀` is chosen so that `dk/dθ` at the initial model equals the sigmoid's `dk/dz_m`; `θ = 0` at the initial model. The first L-BFGS step then matches the Baseline to first order, and differences come from the path. Unmatched variants are a Chain link, because they change the block balance (§2.2 item 5).

**`sat_frac`** must be generalised for non-sigmoid latents. Suggested: the fraction of interior cells with `c` within 1 m/s of a bound or outside it. Report it, together with `c_min` and `c_max`, in both arms [I].

| Rank | One-line change | Hypothesis (metric, against the `SlownessLatent` Baseline) | Falsified if | Cost |
|---|---|---|---|---|
| 1 | `model_latent: raw_k`: Jacobian-matched raw `k` passed straight to the residual, with a soft C¹ box penalty, `c = c₀√k` for reporting, clamped at band handover | Primary: final `rmse_brain` ≥ 10% lower, as the cells trapped at 1300 m/s escape (§3.1). Secondary: `rmse_skull` no more than 5% higher; out-of-box fraction < Baseline `sat_frac` | `rmse_brain` improves by < 10%, or `rmse_skull` worsens by > 5% | S: about 80 lines (latent class, penalty, `model_curvature`'s `dk/dθ`, generalised `sat_frac`, #35's tests) |
| 2 | `model_latent: log_c`: Jacobian-matched `θ = ln(c/c_init)` scaled by `g₀′`, soft box on `c` | As 1 for `rmse_brain`. In a Chain link with `model_preconditioner: pde_diagonal` in both arms: final `rmse_skull` ≥ 10% lower, because `s` stays valid (§3.3) | No `rmse_brain` gain ≥ 10% at link 1; no `rmse_skull` gain ≥ 10% at the `pde_diagonal` link | S: shares candidate 1's code; a third value of the same key |
| 3 | `model_latent: multilevel` (L = 4: 125, 63, 32, 16), each level linearly interpolated and summed, then the current sigmoid map | Primary: final `rmse_skull` ≥ 10% lower. Secondary: `rmse_brain` no more than 10% higher; fringe energy in the water lower in the difference map | `rmse_skull` gain < 10%, or `rmse_brain` > 10% higher | M: about 120 lines; `pde_diagonal` composition; probe-run check of the per-level curvature ratio |
| 4 | `model_latent: coarse_spline` at 4 mm knots (cubic B-spline, `conv_transpose2d`), current sigmoid map on the knots | Primary: final `rmse_brain` ≥ 10% lower (fewer null directions). Secondary: `rmse_skull` no more than 10% higher (rim resolution cost) | `rmse_brain` gain < 10%, or `rmse_skull` > 10% worse | M |
| 5 | `model_latent: raw_k_projected`: raw `k` with a custom projected L-BFGS (active-set freeze, projected line search) | Final `rmse_brain` and `rmse_skull` both within 5% of candidate 1. If so, the soft box suffices and 5 is dropped | Either differs from candidate 1 by > 5% in candidate 5's favour | M–L: custom optimiser, which costs the stock strong-Wolfe wrapper (JBP §6) |
| 6 | `model_latent: siren` (4×64, `ω₀ = 30`, linear output added to the initial model, Jacobian-matched output scale), weights stacked into L-BFGS | Final `rmse_brain` ≥ 10% lower at equal `n_evaluations`, with `rmse_skull` no more than 10% higher | Either threshold missed, or the run fails to reduce the loss below the Baseline's at 50 outer iterations | L |
| 7 | `model_latent: dip_cnn` (Zhu22-style generator, tanh output, added to the initial model) | As 6 | As 6 | L |

Design notes for the Protocols [I]:

- Candidates 1, 2 (and optionally 3) can be **one experiment** with `optimiser.model_latent ∈ {sigmoid_slowness, raw_k, log_c}`: one variable, three values. Report `n_evaluations` and `wall_s` too. Per-evaluation cost should be within noise for 1 and 2.
- **Chain:** (a) repeat the winner with `wavefield_preconditioner: initial_model`, record 0002's better arm; (b) repeat with `pde_diagonal`, which is where candidate 2's staleness argument bites; (c) unmatched scales.
- **Assumptions and limitations** to carry: inverse crime, one phantom, one band, no seeds. The skull is unlikely to reach the range where the sigmoid saturates within 100 outer iterations (§1 result 4), so a null on `rmse_skull` for candidate 1 doesn't rule out a later gain.
- **Context:** record 0002 (bounds hit in the Baseline), #35, #18 §3.4, #97, #98, #111, this file, JBP.

## 7. Open questions and gaps in the literature

- **No ablation of the model variable or bound map in any joint-space Helmholtz study.** WRI and IR-WRI pick squared slowness for bilinearity and don't compare it [S: Aghamiry19 §2.1 states the choice; vLH16 has no comparison]. ODIL's sigmoid has no comparison either [S: Karnakov23].
- **No transcranial study of reparametrisation.** Guasch20 inverts velocity with no stated bounds. MOFI26 and Marty23 use shape or rigid maps that need a template.
- **High contrast and spatial maps:** the salt results are reduced-space and either needed constraints (Esser18) or came out smooth (Zhu22). The 4 mm rim at 2:1 contrast is untested for any spatial map.
- **Bounds:** is `c_max = 3100` (3% above the skull truth) itself a source of trapped cells once the skull is recovered? Widening it is a single-key experiment, independent of the latent [I].
- **`m`-form against `k`-form residual:** which weighting of skull rows the PDE term should use is an ADR-0002 question that this file doesn't answer (§2.1).
- **Interaction with `null(S)`:** whether spatial maps help would be measured directly by JBP §8's `range(S)`/`null(S)` split of the model error, at a multilevel latent.

## 8. Unverified and limits

**Not retrieved in full; only abstract or search excerpts:**

- Esser18, FWIME22, WuMcMechan19, DRFWI25, Marty21, Marty23, Tancik20 (abstract and summary pages). I read P2INR25 through the MICCAI review page only.
- Tarantola's Jeffreys-parameter argument, and Tarantola & Valette 1982, only through a secondary SEP report. Tarantola 1986 only through Carneiro14's citation.
- van den Berg & Abubakar's CSI review: the contrast definition wasn't retrievable (verification wall). The claim that CSI's contrast is affine in squared slowness is my reading of the standard form, not checked against the paper.

**Searched for and not found as primary sources:**

- Virieux & Operto 2009, and Brossier and Operto on parameter choice in frequency-domain FWI: I found the paper record but not its parameterisation text.
- Any FWI study of a periodic, mirror or barrier bound map.
- Any wavelet-parameterised FWI paper with retrievable results.
- Stride (Cueto et al.): parameterisation and bounds not found.
- The Treeby and Aubry groups: no transcranial reparametrisation paper found.

**Limits:**

- §3's tables assume `r = 0`, no source and no PML for the curvature identity, and fixed `|u|`. They are diagonal-only.
- Sensitivities use the default bounds and `z_scale = 1`.
- No candidate was run, piloted or timed here.

## Sources

Retrieved in this session. Read = targeted sections of the full text.

- [vLH13] van Leeuwen T., Herrmann F.J. "Mitigating local minima in full-waveform inversion by expanding the search space." *Geophys. J. Int.* 195:661–667 (2013). Read (eq. 1–7, §2). https://slim.gatech.edu/Publications/Public/Journals/GeophysicalJournalInternational/2013/vanLeeuwen2013GJImlm/vanLeeuwen2013mlm.pdf
- [vLH16] van Leeuwen T., Herrmann F.J. "A penalty method for PDE-constrained optimization in inverse problems." *Inverse Problems* 32:015007. Read through targeted questions. https://arxiv.org/abs/1504.02249
- [Aghamiry19] Aghamiry H.S., Gholami A., Operto S. "Implementing bound constraints and total-variation regularization in extended full-waveform inversion with the alternating direction method of multipliers: application to large contrast media." *Geophys. J. Int.* 218(2):855–872 (2019). Read §1–3.3 (first 100k characters). https://arxiv.org/abs/1902.02744
- [Karnakov23] Karnakov P., Litvinov S., Koumoutsakos P. "Flow reconstruction by multiresolution optimization of a discrete loss with automatic differentiation." arXiv:2303.04679 (*Eur. Phys. J. E*, per search excerpt). Read §2.2, §3.1, §3.4, §3.5. https://arxiv.org/abs/2303.04679
- [Karnakov24] Karnakov P., Litvinov S., Koumoutsakos P. "Solving inverse problems in physics by optimizing a discrete loss." *PNAS Nexus* 3(1):pgae005. Not re-read; see JBP Sources. https://pmc.ncbi.nlm.nih.gov/articles/PMC10799659/
- [Guasch20] Guasch L., Calderón Agudo O., Tang M.-X., Nachev P., Warner M. "Full-waveform inversion imaging of the human brain." *npj Digit. Med.* 3:28 (2020). Read. https://doi.org/10.1038/s41746-020-0240-8
- [Lucka22] Lucka F., Pérez-Liva M., Treeby B.E., Cox B.T. "High resolution 3D ultrasonic breast imaging by time-domain full waveform inversion." *Inverse Problems* 38:025008 (2022). Read §3.3, §4.1, §4.7, §4.8, App. D. https://arxiv.org/abs/2102.00755
- [Carneiro14] Carneiro M.S.R., Bulcão A., Pereira-Dias B., Soares Filho D.M., Duarte F.S., Landau L. "The role of parametrization in acoustic full waveform inversion." VI SimBGf (2014). Read. https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/VI_SimBGf/The%20role%20of%20parametrization%20in%20acoustic%20full%20waveform%20inversion.pdf
- [Esser18] Esser E., Guasch L., van Leeuwen T., Aravkin A.Y., Herrmann F.J. "Total variation regularization strategies in full-waveform inversion." *SIAM J. Imaging Sci.* 11:376–406 (2018). Abstract. https://arxiv.org/abs/1608.06159
- [Zhu22] Zhu W., Xu K., Darve E., Biondi B., Beroza G.C. "Integrating deep neural networks with full-waveform inversion: reparametrization, regularization, and uncertainty quantification." *Geophysics* 87(1):R93–R109 (2022). Read. https://arxiv.org/abs/2012.11149
- [IFWI22] Sun J., Innanen K. "Implicit full waveform inversion with deep neural representation." arXiv:2209.03525 (JGR Solid Earth 2023 per a citation listing, DOI 10.1029/2022JB025964, not confirmed). Read §2–3.2. https://arxiv.org/abs/2209.03525
- [RashtBehesht22] Rasht-Behesht M., Huber C., Shukla K., Karniadakis G.E. "Physics-informed neural networks (PINNs) for wave propagation and full waveform inversions." *JGR Solid Earth* 127(5) (2022), DOI 10.1029/2021JB023120. Read (methods, appendix). https://arxiv.org/abs/2108.12035
- [P2INR25] Wang Z., Yan W., Liu Z., Yuchi M., Qiu W. "P2INR-FWI: an implicit neural representation method for speed of sound image reconstruction in ultrasound computed tomography." MICCAI 2025. Review page only. https://papers.miccai.org/miccai-2025/0662-Paper2163.html
- [WuMcMechan19] Wu Y., McMechan G.A. "Parametric convolutional neural network-domain full-waveform inversion." *Geophysics* 84(6):R881–R896 (2019). Abstract. https://doi.org/10.1190/geo2018-0224.1
- [DRFWI25] Liu F., Li Y., Su R., Huang J., Bai L. "Deep reparameterization for full waveform inversion: architecture benchmarking, robust inversion, and multiphysics extension." arXiv:2504.17375. Abstract. https://arxiv.org/abs/2504.17375
- [FWIME22] "Full waveform inversion by model extension: theory, design and optimization." arXiv:2205.14341. Search excerpt. https://arxiv.org/abs/2205.14341
- [Tancik20] Tancik M. et al. "Fourier features let networks learn high frequency functions in low dimensional domains." NeurIPS 2020. Abstract and summary. https://arxiv.org/abs/2006.10739
- [MOFI26] Bates O., Cueto C., Coleman C., Smith C.A.B., Guasch L., Calderón Agudo O. "Automatic skull-template alignment without a guidance image." arXiv:2601.14533 (2026). Read through a summary of the HTML. https://arxiv.org/abs/2601.14533
- [Marty21] Marty P., Boehm C., Fichtner A. "Acoustoelastic full-waveform inversion for transcranial ultrasound computed tomography." Proc. SPIE 11602 (2021). Abstract. https://doi.org/10.1117/12.2581029
- [Marty23] Marty P., Boehm C., Fichtner A. "Shape optimization for transcranial ultrasound computed tomography." Proc. SPIE 12470 (2023). Abstract. https://doi.org/10.1117/12.2654328
- [SEP143] Stanford Exploration Project report 143 (Sjoerd de Ridder), secondary account of Tarantola (2006) on Jeffreys quantities and Tarantola & Valette (1982). Search excerpt. https://sepwww.stanford.edu/data/media/public/docs/sep143/sjoerd1/paper_html/node4.html
- [torchLBFGS], [JBP]: see [joint-block-preconditioning.md](joint-block-preconditioning.md), Sources.
