# Absorbing boundaries for the FD Helmholtz operator, and the velocity inside the PML

Question (#134, absorbing #91): which absorbing layer should the 2D finite-difference Helmholtz operator use, how thin can it be, and what should the velocity inside the PML be? Scope: `Sponge` (`odil_wave/operator/conditions.py`), `WaveEquation` (`operator/utils.py`), `HelmholtzSolver.assemble_H_sparse` and `laplacian_csr` (`operator/helmholtz.py`), `Grid._build_pml_profiles` / `_sigma_max` (`grid/grid.py`), `VelocityModel.pml_c` / `build_full_c` (`models/velocity_models.py`), `JointFreqODIL` (`optimisation/joint_odil.py`), ADR-0002, ADR-0003, #18 §3.5, #37, #111 H5.

Domain terms follow `GLOSSARY.md`: **PML**, **Interior**, **PDE term**, **Data term**, **Wavefield block**, **Model block**, **Wavefield preconditioner**, **Inverse crime**, **Same physical problem**. `s_ξ = 1 + σ_ξ/(iω)` is the complex coordinate stretch along axis `ξ` (the repo's `e^{iωt}` convention: `λ_t ≈ iω`, `λ_tt ≈ −ω²`).

Every claim carries one provenance tag:

- **[S]**: established by a source. The key points to Sources. "(abstract)" means I reached only the abstract or a search excerpt. "(via X)" means I read the claim only as X reports it.
- **[R]**: read from this repo's code, an issue or an experiment record.
- **[P]**: measured in a scratch pilot in this session (float64 SciPy, standalone 2D operator, not the repo's code; §6).
- **[I]**: my own inference or derivation.

## 1. Conclusion

1. **The current layer is a damped wave equation, not a PML, and it is strongly frequency-dependent.** Its diagonal `λ_tt + σ_Σλ_t + σ_Π` is exactly `−ω² s_x s_y`, but the Laplacian is unstretched [R: `helmholtz.py`, `conditions.py`; #18 §3.5]. That is the time-domain sponge `u_tt + σ u_t = c²Δu`, the same form Devito's examples use (`m u_tt − Δu + damp u_t`) [S: Devito `acoustic/operators.py`]. In the pilot it reaches a 7e-5 relative interior error at the default (40 cells, 80 kHz), but 1.9e-3 at 40 kHz and 8.7e-2 at 20 kHz with the same 40 cells [P]. Today's 63% PML share buys acceptable absorption only at the top band.
2. **A conservative stretched-coordinate PML is 3–4× thinner for the same error, at every band tested.** With the Laplacian replaced by `s_y ∂_x((1/s_x)∂_x) + s_x ∂_y((1/s_y)∂_y)` and the repo's own cubic profile and `σ_max` formula, 10 cells give 5.5e-5 to 1.7e-4 and 15 cells 1.0e-5 to 2.7e-5 across 20–120 kHz (6–37 points per wavelength); 12 cells give 3.7e-5 at 80 kHz [P]. In cells, not wavelengths, is what sets the width of a stretched PML here [P].
3. **The discretisation of the stretched Laplacian matters more than the profile.** The conservative (flux) form with `1/s` on the half-nodes beats the expanded form `(1/s)∂²u − (s'/s²)∂u` (PySIT's "compact" operator [S: PySIT]) by 10–100× at equal width. Bolting a 2nd-order stretch correction onto an order-4 interior stencil is just as bad [P]. For order > 2 the conservative form needs staggered order-P first derivatives everywhere, which changes the interior stencil (§4.2).
4. **Bermúdez's unbounded profile doesn't carry over to a node-sampled FD grid.** Its absorption comes from the integral of `σ = c/(L − d)` diverging at the outer wall [S: Bermudez07 (abstract)]. Sampled at nodes, that integral is finite, so the round-trip decay is only about `1/(p+1)²`: 2.8e-2 at 10 cells, worse than the polynomial layer at every width [P]. Modave et al. report the shifted hyperbolic profile as efficient for FD, FV and FE schemes in the time domain [S: Modave14 (abstract)]. I couldn't read how they place the singularity, so this conflict is unresolved (§7).
5. **A frozen `pml_c` that differs from the interior edge reflects far more than any PML.** With a 15-cell stretched PML, a 50 m/s mismatch between the interior edge and `pml_c` gives a 3–4% interior error, against 1.5e-5 with no mismatch, about 2000× more [P]. That matches the normal-incidence Fresnel coefficient `|Δc|/(c₁+c₂)` [I]. k-Wave, Devito, Stride and JUDI all extend the medium into the absorbing layer by **replicating the edge values** [S: kWave `expandMatrix`; Devito `initialize_function`; Stride `pad_data`; JUDI]. JUDI optionally sums the padding's gradient onto the edge cells as the exact adjoint [S: JUDI `remove_padding`].

### Recommendations

| # | What | For | Evidence |
|---|---|---|---|
| 1 | **Form:** `−ω² s_x s_y u − c²[s_y D⁻_x((1/s_x^{½}) D⁺_x u) + s_x D⁻_y((1/s_y^{½}) D⁺_y u)] = s_x s_y f`, with `s` on the half-nodes inside the brackets. Keep the cubic profile and `_sigma_max`. `damping_scale` must scale `σ` inside every `s`, not only the mass term. | #37 | §3, §6 [P] |
| 2 | **Width:** 12–15 cells at order 2 (15 cells: ≤ 2.7e-5 from 20 to 120 kHz at today's grid; 12 cells: 3.7e-5 at 80 kHz). Measure the final number in #37's validation before changing the default. | #37, #138 | §6.1 [P] |
| 3 | **Validation:** thick-layer reference (same scheme, 150 cells, `R0 = 1e-12`), relative L2 error over the interior and at the receivers, against width, at the lowest and highest band frequency, with a source 30 mm from the interface and one near a corner. Don't use the analytic Hankel solution as the reference. | #37 | §5 |
| 4 | **PML velocity:** replicate the interior edge into the PML (`F.pad(..., mode="replicate")` in `build_full_c`) and let autograd carry the gradient back onto the edge cells. Drop `pml_c` as a frozen constant. If that is rejected, freeze a strip of interior cells next to the PML at `pml_c`. | #91, #111 H5 | §4.4 |
| 5 | **Order > 2 (#138):** conservative staggered order-P first derivatives everywhere (`D⁻ diag(1/s_h) D⁺`). Order 4 gives 7e-5 at 10 cells, the same as order 2 [P]. The interior stencil becomes the 7-point staggered-squared one rather than `FD2_COEFFS[4]`. | #37, #138 | §4.2 [P] |
| 6 | **Joint-solve conditioning:** before trusting thin layers with the wavefield preconditioner off, check the PML rows' scale (§4.3). Weighting PML rows by `1/(s_x s_y)` is the candidate fix. | new ticket | §4.3 [I] |

## 2. Where the code stands

- **Operator** [R]: `H = λ_tt + w(σ_Σ λ_t + σ_Π) − c'² L`, with `L` the order-P cross stencil with reflect (Neumann-mirror) padding. The sparse (`assemble_H_sparse`) and matrix-free (`WaveEquation.residual`) forms are the same matrix. `L` is frequency-independent and cached per grid (`laplacian_csr`).
- **Profile** [R]: `σ(d) = σ_max (d/L)^3`, `σ_max = (p+1) c_max ln(1/R0) / (2L)` with `c_max = 3100` (the box bound, not the PML speed), `R0 = 1e-6`, 40 cells. `R0` is the continuous-PML design formula, so it doesn't predict this layer's reflection [R: #18 §3.5].
- **Size** [R]: 125² interior at `h = 2.016 mm`, so the default grid is 205² and `N_int/N = 0.372`. At 80 kHz in water, λ = 18.75 mm = 9.3 cells, and the PML is 4.3 λ.
- **Velocity** [R]: the PML is filled with the constant `pml_c` (`build_full_c`, `F.pad(mode="constant")`): the interior background for skull profiles, else the field minimum. The model latent covers interior cells only. The regulariser ignores the PML.
- **Joint objective** [R]: the PDE term sums over interior and PML rows and is normalised by `N_int` (ADR-0002). PML rows constrain only `u`, since `c` is frozen there.
- **Warning** [R]: `wavefield/base.py` warns when the PML is thinner than half the longest wavelength (`c_max/f_min/2`). That is a damping-layer rule, and §6.1 shows it doesn't describe a stretched PML.

## 3. Stretched-coordinate PML forms for FD Helmholtz

- **Origin.** Bérenger's split-field PML [S: Berenger94 (abstract)] is equivalent to a complex change of coordinates `x̃ = x + (1/(iω))∫σ`, so `∂_x → (1/s_x)∂_x` [S: ChewWeedon94 (abstract); CollinoMonk98 (abstract)]. Multiplying by `s_x s_y` gives the 2D frequency-domain form in Recommendation 1, which is #18 §3.5's equation [R]. In 2D, `∂_x((s_y/s_x)∂_x) = s_y ∂_x((1/s_x)∂_x)`, since `s_y` doesn't depend on `x`, so each axis stays a 1D operator scaled by a diagonal [I].
- **Why the current layer reflects.** In a 1D strip the sponge's wavenumber is `k√s` and its impedance changes with `σ/ω`, whereas the PML's is `k s` with matched impedance [I]. The mismatch grows as `σ/ω` grows, which is why the sponge gets worse at low frequency at a fixed `σ` (§6.1) [I, P].
- **Discrete reflection.** A discretised PML is not reflectionless. The residual reflection depends on the incidence angle and on how fast `σ` grows per cell [S: Chern19, citing Bérenger]. Chern constructs a discrete PML with no numerical reflection at all for the 2nd-order scheme, and Hojas et al. extend it to arbitrary order [S: Chern19; Hojas24 (abstract)]. Both are formulated for the time-domain wave equation with auxiliary fields. They are a fallback if Recommendation 1's few-1e-5 error isn't enough (§7) [I].
- **CFS-PML.** `s = κ + σ/(α + iω)` adds a frequency shift `α` that switches the damping off at low frequency. It was introduced to absorb grazing and evanescent waves in time-domain codes [S: KomatitschMartin07 (via later papers)]. Stride's CFS-PML uses `α ∝ 0.01π f_centre` with a quadratic profile [S: Stride `boundaries/devito.py`]. In a single-frequency solve, `α` only rescales `σ/ω`. I found no reason to need it for a 2D square domain whose sources sit 30 mm or more from the layer [I].
- **Profiles.** Polynomial `(d/L)^n`, `n = 2–4`, is the standard choice. k-Wave uses `n = 4` with `σ_max = α c_ref/Δx` (default 2 Np per cell, 20 cells in 2D), where `c_ref` defaults to the maximum sound speed [S: kWave `getPML.m`, `kspaceFirstOrder_setSoundSpeedRef.m`, PML docs]. Too large an `α` reflects from the layer itself [S: kWave PML docs]. Devito and Stride use a sine ramp `pos − sin(2π pos)/(2π)` for their sponges [S: Devito `model.py`; Stride `boundary.py`]. In the pilot, cubic with the repo's `σ_max` is as good as cubic designed at the true speed (§6.1) [P].

## 4. Fit with this repo

### 4.1 Sparse and matrix-free operators

- `s` depends on `ω`, so the stretched Laplacian is per frequency. `laplacian_csr`'s cache would hold the static `D⁺` matrices, with `diag(1/s_h)` applied per frequency. `assemble_H_sparse` is already per frequency [R, I].
- For order 2, `D⁻ diag(1/s_h) D⁺` has the same 5-point pattern as today, so `nnz` and the SuperLU fill are unchanged [I]. `−D⁺ᵀD⁺` has natural Neumann ends, which matches the reflect padding [I].
- Matrix-free: forward difference (conv), multiply by a complex `(nf, nx−1, ny)` face coefficient, backward difference (conv), per axis. The existing real-kernel conv on the real and imaginary parts still works, because the complex multiply happens between the two convs [I].
- **Interior rows.** With `σ` sampled at the half-node `d = h/2`, the last interior row sees `1/s_h ≠ 1` by `σ_max(1/2p)³/ω`, about 5e-4 at 12 cells and 80 kHz [I]. To keep #37's "interior physics unchanged" literally true, start the profile at the first PML face, i.e. measure `d` from `h/2` past the last interior node [I].
- **Symbols.** Today's diagonal uses the leapfrog symbols, and `λ_tt ≠ λ_t²`, so it isn't exactly a product `λ_tt s_x s_y` [R, I]. Any consistent choice is a PML as long as the mass term and the Laplacian use the same `s`. The simplest is `s = 1 + σ/(iω)` with the exact `ω`, and mass `λ_tt s_x s_y`. #18 §3.6's switch to exact symbols makes this moot [I].

### 4.2 Higher-order stencils

- **Conservative staggered form.** The 4th-order staggered derivative `(9/8, −1/24)` composed as `−D⁺ᵀ diag(1/s_h) D⁺` reflects 7.4e-5 at 10 cells and 1.4e-5 at 15 cells (80 kHz), the same as order 2 [P]. In the interior it is a 7-point 4th-order stencil, not the 5-point `FD2_COEFFS[4]`. #138's "order 4" would mean this stencil [I].
- **Rejected:** the expanded form with centred order-4 `D_x`: 3.5e-3 at 10 cells, 80 kHz, and 1.7e-2 at 20 kHz. Also rejected: the order-4 interior stencil plus a 2nd-order conservative stretch correction `D⁻ diag(1/s_h − 1) D⁺`: 2.6e-3 and 1.6e-2 [P]. Both mix discretisations whose difference is O((kh)²) and gets multiplied by a large `1/s` [I].

### 4.3 PML rows in the joint objective

- **Literature gap.** I found no source on L-BFGS over a joint (ODIL or WRI-penalty) objective with PML rows. WRI and penalty codes penalise the full discrete operator, absorbing rows included, as SimpleFWI's `getA` builds its ABC into `A` [S: SimpleFWI `getA.m`] [I]. JBP's gap stands for both the sponge and the stretched PML [I].
- **Row scale.** With the repo's `σ_max`, `σ_max/ω` is 2.1 at 40 cells and 7.0 at 12 cells at 80 kHz, and 8.4 and 28 at 20 kHz [I, from `_sigma_max`]. The mass coefficient `|ω² s_x s_y|` in a corner is about `(σ_max/ω)²` times the interior's: about 50 at 12 cells and 80 kHz, about 800 at 20 kHz. The corner rows of `AᴴA` then scale like its square [I]. A thin layer therefore spreads the wavefield block's diagonal more than today's thick one. That doesn't matter for the factor solve or for the `A(c₀)⁻¹` wavefield preconditioner, which contains the PML exactly [I, ADR-0003]. It could matter for plain L-BFGS.
- **Candidate fix.** Divide the PML rows by `s_x s_y`, i.e. use `−ω²u − c²[(1/s_x)∂_x((1/s_x)∂_x u) + …]` there. It has the same solution, a unit mass coefficient and Laplacian coefficients ≤ 1 in magnitude. It is a fixed row weighting confined to rows where `c` is not inverted [I]. ADR-0003 already expects a diagonal to "even out the PML" [R]. This needs a measurement (Recommendation 6).
- **Balance.** ADR-0002's normalisation by `N_int` is unaffected. A thinner PML lowers the PML's share of the PDE term, and `N_int/N` rises from 0.372 to 0.70 at 12 cells [I].
- **Cost.** 205² → 149² unknowns (1.9× fewer). 2D nested-dissection factorisation scales like `n^{1.5}`, so the factor is about 2.6× cheaper. L-BFGS vectors are 1.9× smaller [I].

### 4.4 Velocity inside the PML

Three options:

- **(a) Constant `pml_c` (today).** It is exact only while the interior edge stays at `pml_c`. The truth's edge is background, and the homogeneous initial model is 1500 m/s [R]. Once the inversion moves an edge cell, the jump reflects `|Δc|/(2c)`: 1.7% at 50 m/s, 7% at 1300 vs 1500 m/s. The pilot error is 3.9e-2 and 2.0e-1 respectively [P, I]. #111 records "a low-velocity band along and outside the transducer ellipse, plus ringing out to the PML" [R: #111]. A drifted edge against a frozen `pml_c` is a mechanism for H5 that doesn't need any PML reflection [I]. Check: plot the edge cells' `c` against `pml_c` in #111's run.
- **(b) Edge-extended (what the codes do).** k-Wave's `expandMatrix` "enlarges an input matrix by extension of the values at the outer faces" when the PML sits outside the grid [S: kWave `expandMatrix.m`, `kspaceFirstOrder_expandGridMatrices.m`]. Devito's `initialize_function` fills the padding "by repeating its nearest boundary plane" [S: Devito]. Stride pads `vp` with `np.pad(mode='edge')` and scales its damping by the local velocity [S: Stride `data.py`, `boundary.py`]. For gradients, Stride and JUDI crop the padding by default; JUDI's `sum_padding=true` adds it onto the boundary rows "for true adjoints" [S: Stride `iso_acoustic/devito.py`; JUDI `remove_padding`, `OptionsStructure.jl`].
  - Why it works: the extended medium is invariant along the layer's normal by construction, which is the case in which a PML is matched [I].
  - In torch, `F.pad(mode="replicate")` gives JUDI's true-adjoint sum automatically [I].
  - Cost: each edge cell's gradient and curvature collect up to `p` PML cells. `model_curvature` (`pde_diagonal`) computes `H_kk` on interior cells only, so it would miss that contribution [R, I].
- **(c) Inverted inside the PML.** None of the four codes does it [S: as above]. The field decays there, so `c` is weakly identifiable, and the layer's matching assumes a known medium [I]. Not recommended.

Recommendation: (b). A fallback is (a) plus a frozen interior strip a few cells wide, which removes the jump without coupling PML rows to the latent [I].

## 5. Validation method for #37

- **Reference:** the same scheme with a very thick layer: 150 cells, cubic, `R0 = 1e-12`. In the pilot it agrees with 200 cells to 1e-9 at every frequency, well below the 1e-5 to 1e-4 being measured [P].
- **Not the Hankel function.** The analytic 2D Green's function `−(i/4)H₀⁽²⁾(kr)` measures numerical dispersion as well as reflection. At 9.3 points per wavelength the 2nd-order phase error over the 250 mm interior is far above 1e-4, so it can't resolve PML reflections [I]. Use it only to sanity-check the reference.
- **Also not the hyperbolic layer.** A first pilot reference using the shifted-hyperbolic layer had a 2.9e-5 floor of its own [P].
- **Metric:** relative L2 error of the interior field, and of the receiver traces (the data-space number that matters for the inverse crime), against width {5, 8, 10, 12, 15, 20, 40}.
- **Cases:** frequencies at both ends of the band set (today 20–120 kHz covers it); a source 30 mm from the interface (the transducer ring's clearance); a source near a corner (grazing incidence on two faces); homogeneous water.
- **Tests:** (i) sparse and matrix-free residuals agree, as today; (ii) interior rows unchanged when the profile starts at the first PML face (§4.1); (iii) a small-grid reflection-vs-width test in the suite, asserting monotone decay and a threshold at the chosen width.
- **Side effect:** the default PML split (observation PML 70 vs inversion 40, 3.5e-4 misfit at the truth [R: #18 §3.6]) will shrink by orders of magnitude. It stops being a usable modelling error [I].

## 6. Pilot

Standalone 2D operator: 125² interior at `h = 0.25/124 m`, homogeneous `c = 1500 m/s`, Gaussian source (σ = 2.5 mm) at (30, 125) mm, i.e. 30 mm from the left interface. Exact `iω`/`−ω²`, SciPy `splu`, float64. "Error" is `‖u_p − u_ref‖/‖u_ref‖` over the interior, with `u_ref` from the conservative stretched PML at 200 cells, `R0 = 1e-12`. "repo" is `σ_max` from `_sigma_max` (cubic, `c_max = 3100`, `R0 = 1e-6`). The scripts were scratch files outside the repo and aren't committed. Differences from the repo's operator: the sponge column uses Dirichlet ends instead of the mirror, and exact rather than leapfrog symbols.

### 6.1 Reflection against width, order 2

| cells | 80 kHz sponge (today) | 80 kHz PML (repo σ) | 40 kHz sponge | 40 kHz PML | 20 kHz sponge | 20 kHz PML | 120 kHz sponge | 120 kHz PML |
|---|---|---|---|---|---|---|---|---|
| 5 | 4.3e-1 | 5.9e-3 | 8.5e-1 | 2.8e-3 | 1.5 | 3.4e-3 | 2.3e-1 | 1.8e-2 |
| 10 | 1.1e-1 | 7.6e-5 | 3.9e-1 | 5.5e-5 | 9.2e-1 | 6.3e-5 | 2.8e-2 | 1.7e-4 |
| 12 | 6.1e-2 | 3.7e-5 | | | | | | |
| 15 | 2.3e-2 | 1.5e-5 | 2.2e-1 | 1.1e-5 | 6.2e-1 | 1.0e-5 | 3.1e-3 | 2.7e-5 |
| 20 | 4.0e-3 | 4.8e-6 | 1.0e-1 | 3.4e-6 | 4.1e-1 | 3.2e-6 | 2.8e-4 | 8.4e-6 |
| 40 | 7.4e-5 | 3.0e-7 | 1.9e-3 | 2.1e-7 | 8.7e-2 | 2.0e-7 | 1.9e-5 | 5.3e-7 |

Points per wavelength: 9.3 (80 kHz), 18.6 (40), 37.2 (20), 6.2 (120). Other variants at 80 kHz and 10/15/20 cells:

- Cubic designed at `c` with `R0 = 1e-6`: 3.9e-5 / 8.7e-6 / 3.3e-6.
- Quadratic: 2.7e-4 / 7.9e-5 / 3.5e-5.
- Shifted hyperbolic: 2.8e-2 / 1.2e-2 / 6.4e-3.
- Expanded (non-conservative) form, repo σ: 1.0e-2 / 1.2e-3 / 2.4e-4 [P].

### 6.2 Order 4, 10 / 15 / 20 cells

| form | 80 kHz | 20 kHz |
|---|---|---|
| staggered conservative | 7.4e-5 / 1.4e-5 / 4.6e-6 | 6.2e-5 / 1.0e-5 / 3.3e-6 |
| expanded, centred order-4 `D_x` | 3.5e-3 / 2.4e-4 / 9.0e-5 | 1.7e-2 / 4.5e-3 / 1.8e-3 |
| `FD2_COEFFS[4]` + 2nd-order stretch correction | 2.6e-3 / 2.4e-4 / 3.8e-5 | 1.6e-2 / 5.1e-3 / 2.1e-3 |
| sponge (today) | 1.1e-1 / 2.3e-2 / 3.9e-3 | 9.3e-1 / 6.2e-1 / 4.1e-1 |

### 6.3 PML velocity, 15-cell conservative PML, 80 kHz

| interior `c` | `pml_c = 1500` frozen | edge-extended | Fresnel `|R|` |
|---|---|---|---|
| 1300 | 2.0e-1 | 2.0e-5 | 7.1e-2 |
| 1450 | 3.9e-2 | 1.6e-5 | 1.7e-2 |
| 1500 | 1.5e-5 | 1.5e-5 | 0 |
| 1550 | 3.4e-2 | 1.4e-5 | 1.6e-2 |

## 7. Open questions and unverified

- **Hyperbolic profile.** Modave et al.'s finding that the shifted hyperbolic profile works for FD conflicts with §6.1. I read only their abstract. Their placement of the singularity (between nodes, or with a Dirichlet wall at it) may explain the difference [S: Modave14 (abstract)].
- **Joint-solve conditioning (Recommendation 6).** Is L-BFGS on the wavefield block with the wavefield preconditioner off slower with a 12-cell PML than with today's 40 cells, per unit of work? Does the `1/(s_x s_y)` row weighting fix it? Unmeasured. Candidate new ticket.
- **Edge extension with `pde_diagonal`.** With edge extension, `model_curvature` has to include the PML cells' contribution to the edge cells. Unmeasured.
- **Discrete-holomorphic PML** [S: Chern19; Hojas24 (abstract)]: needed only if a tolerance below about 1e-5 at 10 cells is ever required. Not checked for the frequency-domain form.
- **Unread primary sources.** I couldn't reach Operto et al.'s FDFD PML details (HAL bot wall), the k-Wave site (connection refused; read via a mirror and the source), or the full Bermúdez paper.
- **Pilot coverage.** The pilot covers a homogeneous medium only. A skull near the interface is not modelled; the transducer ring keeps it 30 mm or more away [R].

## Sources

- [Berenger94] Bérenger J.-P. "A perfectly matched layer for the absorption of electromagnetic waves." *J. Comput. Phys.* 114(2):185–200 (1994). Not read; standard citation.
- [ChewWeedon94] Chew W.C., Weedon W.H. "A 3D perfectly matched medium from modified Maxwell's equations with stretched coordinates." *Microw. Opt. Technol. Lett.* 7(13):599–604 (1994). Not read; standard citation.
- [CollinoMonk98] Collino F., Monk P. "The perfectly matched layer in curvilinear coordinates." *SIAM J. Sci. Comput.* 19(6):2061–2090 (1998). Not read; standard citation.
- [Bermudez07] Bermúdez A., Hervella-Nieto L., Prieto A., Rodríguez R. "An optimal perfectly matched layer with unbounded absorbing function for time-harmonic acoustic scattering problems." *J. Comput. Phys.* 223(2):469–488 (2007). Abstract. https://doi.org/10.1016/j.jcp.2006.09.018
- [Modave14] Modave A., Delhez E., Geuzaine C. "Optimizing perfectly matched layers in discrete contexts." *Int. J. Numer. Methods Eng.* 99(6):410–437 (2014). Abstract. https://doi.org/10.1002/nme.4690
- [Chern19] Chern A. "A reflectionless discrete perfectly matched layer." *J. Comput. Phys.* 381:91–109 (2019). Read §1 and the results summary. https://arxiv.org/abs/1804.01390
- [Hojas24] Hojas V., Pérez-Arancibia C., Sánchez M. "Reflectionless discrete perfectly matched layers for higher-order finite difference schemes." *SIAM J. Sci. Comput.* 46(5):A3094–A3123 (2024). Abstract. https://arxiv.org/abs/2306.13189
- [KomatitschMartin07] Komatitsch D., Martin R. "An unsplit convolutional perfectly matched layer improved at grazing incidence for the seismic wave equation." *Geophysics* 72(5):SM155–SM167 (2007). Via later papers.
- [kWave] ucl-bug/k-wave at `94e5583`: `k-Wave/private/getPML.m`, `k-Wave/expandMatrix.m`, `k-Wave/private/kspaceFirstOrder_expandGridMatrices.m` L115–135, `k-Wave/private/kspaceFirstOrder_setSoundSpeedRef.m`, `k-Wave/kspaceFirstOrder3D.m` L617–622. Read. https://github.com/ucl-bug/k-wave. "Controlling the PML" example page, read via a mirror: https://raw.githubusercontent.com/jdesbonnet/experiments_with_k-Wave/master/helpfiles/example_na_controlling_the_pml.html
- [Devito] devitocodes/devito at `35e25b3`: `examples/seismic/model.py` (`initialize_damp`, `nbl=20`), `devito/builtins/initializers.py` (`initialize_function`, mode `'constant'` repeats the boundary plane), `examples/seismic/acoustic/operators.py` L74, L99. Read. https://github.com/devitocodes/devito
- [Stride] trustimaging/stride at `36c10ac`: `stride/physics/boundaries/devito.py` (`SpongeBoundary2`, `ComplexFrequencyShiftPML2`), `stride/physics/boundaries/boundary.py` (`damping`), `stride/problem/data.py` L513–536 (`pad_data`, `mode='edge'`), `stride/physics/iso_acoustic/devito.py` L128, L1072. Read. https://github.com/trustimaging/stride
- [JUDI] slimgroup/JUDI.jl at `655cc44`: `src/TimeModeling/Utils/auxiliaryFunctions.jl` L79–89 (`remove_padding`), `src/TimeModeling/Types/OptionsStructure.jl` L66 (`sum_padding`), `src/TimeModeling/Types/ModelStructure.jl` L325 (`nb=40`), `src/TimeModeling/Modeling/misfit_fg.jl` L77. Read. https://github.com/slimgroup/JUDI.jl
- [PySIT] pysit/pysit at `c6de681`: `pysit/solvers/constant_density_acoustic/frequency/constant_density_acoustic_frequency_scalar_base.py` L147–179 (compact PML operator), `…_scalar_2D.py` L52–103. Read. https://github.com/pysit/pysit
- [SimpleFWI] TristanvanLeeuwen/SimpleFWI at `6990617`: `core/getA.m`. Read. https://github.com/TristanvanLeeuwen/SimpleFWI
- Repo: #18 §3.5–3.6, #37, #91, #111, ADR-0002, ADR-0003, [joint-block-preconditioning.md](joint-block-preconditioning.md) (JBP).
