# Preconditioning the joint wavefield and model blocks on a GPU

Question: which preconditioning methods could run on a GPU and improve the conditioning of the joint (wavefield + model) Hessian in this repo's frequency-domain ODIL inversion? Scope: ADR-0002 and ADR-0003, #96, #93, #18, #33, #111, #98, #104, record 0002.

Domain terms follow `GLOSSARY.md`: **Wavefield block**, **Model block**, **Wavefield preconditioner**, **Model preconditioner**, **Initial model**, **Warm start**, **True model**, **Interior**, **PML**, **Same physical problem**, **Scratch run**.

Every claim carries one provenance tag:

- **[S]**: established by a source. The key points to Sources. "(abstract)" means I reached only the abstract or a search excerpt.
- **[R]**: measured in this repo, in an issue or an experiment record.
- **[P]**: measured in a scratch pilot in this session: float64 linear algebra on uncommitted code, with no L-BFGS. It indicates, it does not establish. Under the glossary's rule for scratch runs, no experiment record may cite it.
- **[I]**: my own inference or derivation. Nothing in the literature backs it for this regime unless a tag says so.

## 1. Conclusion and ranked recommendation

Three results decide the ranking. None is in the handoff.

1. **Block-diagonal preconditioning can't fix the joint Hessian, even when exact.** With the data term off, `H_uu⁻¹ᐟ²H_uzH_zz⁻¹ᐟ²` has singular values of exactly 1, and the Schur complement `S` of the model block is exactly 0 [I, P]. Each model direction has an equally cheap wavefield direction that undoes it. Under exact block scaling the spectrum is `{1} ∪ {1 ± √(1−μᵢ)}`, with `μ` the eigenvalues of `H_zz⁻¹ᐟ²SH_zz⁻¹ᐟ²`. Only the data term lifts `μ` off 0. On the default problem `diag(S)/H_zz` has a median of 0.32% at 2 mm and 0.08% at 1 mm, so it falls like `h²` [P].
2. **The data term sets the top of the preconditioned wavefield block.** On the default problem `A(c₀)⁻¹` leaves 63 outliers per shot, between 591× and 4746× the bulk [P]. In van Leeuwen & Herrmann's scale the penalty is `λ̃ ≈ 2e-4` [S for the scale, P for the value], far below their "small" threshold of 1. This puts a number on #33's "`u` absorbs the misfit". Both numbers are unchanged at 1 mm (569–4778, `λ̃` 2.1e-4), which agrees with ADR-0002's grid independence [P].
3. **#96's flat iteration count doesn't carry over to the skull.** `κ(H_v)` for the PDE term with `A(c₀)⁻¹` is 2.2 for #96's 8% overdensity, but 1.16e5 for the default true model [P]. Against that, it is nearly flat in resolution at the default true model (1.16e5 at 2 mm, 1.57e5 at 1 mm [P]). `#105` ran with this stale spectrum.

The literature has **no evidence** for this exact regime: L-BFGS on a normal-equations wavefield block, trapping skull geometry, and a graded damping layer rather than a stretched PML. Rankings below rest on the pilot and on inference where marked. None has been through ADR-0003's test.

### Ranked plan

| # | What | Why first or later | Form | Evidence |
|---|---|---|---|---|
| 0 | **Run the joint-spectrum measurement** (§8). The small-grid version was piloted here; the 205² matrix-free one takes hours. | Decides between items 1 to 6. Costs no optimiser runs except the error projection. | n/a | P |
| 1 | **Shift the factors**: use `C = A(c₀) + iβ\|λ_tt\|`, `β ≈ 0.1–0.2`, in `FactorSolve` in place of `A(c₀)`. | Cheapest possible test: one extra factorisation per band (ADR-0003's wording would need amending, §7), same apply cost. κ at the true model falls from 1.16e5 to 1.9–2.2e4 [P]. It gives up exactness at `c₀` (κ 28–95 against 1). | factored | P; shifted-operator theory [S] for milder problems |
| 2 | **Data-term Woodbury factor**: `M = B·(I + τVᴴV)⁻¹ᐟ²`, `V = PB`. | Removes the outliers in result 2. Never goes stale in `c`, because the data term has no `c`. Pilot on a small grid: κ⁺ falls 9× to 400× at every state tried, including high contrast [P]. | factored | P |
| 3 | **GPU-native exact or shifted inverse**: separable fast diagonalisation for a homogeneous initial model, dense block-tridiagonal LDU (or cuDSS on CUDA) otherwise. | Removes the CPU solves, 72% of `optimise_s` in record 0002 [R]. Doesn't change the conditioning of its exact counterpart. | factored | P, cost arithmetic |
| 4 | **Refresh by splitting the band**, or by mapping L-BFGS history into the new coordinates (§6). | Factors cost 0.14–0.38 s against about 10 s per outer step [P, R], so refresh is cheap. But refreshing at record 0002's final model only cuts κ 10× (1.16e5 to 1.10e4 [P]), so it's bounded until the skull moves. | factored, with history mapping or restart | P, I |
| 5 | **Shear plus `diag(S)` model scale**: `ũ = B(v − J₀ s y)` with `s` from `diag(S)`. | The only candidate that addresses result 1. Fresh on a small grid: κ⁺ 1.8e4 to 156 [P]. But it loses that once `c` moves: at a 2000 m/s ring, 8.5e3 against 4.8e3 for block-diagonal plus Woodbury; at 3000 m/s, 44× worse [P]. Worth testing for small model updates (later bands, #98's cases), not for the first skull band. | factored | P, I |
| 6 | **Custom L-BFGS with an `H0` hook.** | Gives per-block `γ` and non-factored `H0`. Equivalent to the change of variables otherwise (§6), so only worth building if items 1 to 5 need it. | SPD `H0` | S (variable `H0`), I |

### Ruled out, and why

- **Jacobi on the wavefield block:** gain fades to 1.3× and iterations still grow `h⁻²` [R: #96]. On the default true model κ stays 1.5e7 [P].
- **ILU, SPAI or FSAI, algebraic multigrid as a GPU wavefield preconditioner:** classical methods fail on the indefinite operator [S: ErnstGander12 (abstract)]; ILU needed fill level 21 to be useful in the only full-space Helmholtz study [S: Malovichko22]; AmgX lists complex support as "currently limited" [S: AmgX]; I found no statement of complex support in Ginkgo's preconditioner docs [S: Ginkgo].
- **Constant-coefficient DCT or FFT shifted inverse without the PML:** κ at `c₀` is 2.8e4–6.6e4 against 28–83 for versions that include the PML [P]. The separable form includes the PML at a modest extra cost, so it dominates. At the true model the DCT form ties `A(c₀)⁻¹` (1.1e5) but no better.
- **Any inner Krylov solve inside `M`** (GMRES, CG to tolerance, flexible schemes): this makes `M` nonlinear and variable, which neither a fixed change of variables nor stock L-BFGS allows [I].
- **Learned preconditioners** (for now): nonlinear in the right-hand side, used inside FGMRES [S: Azulay22, Lerer24], trained on natural or geophysical images, and needing training data. No evidence on skull geometry or normal equations.
- **WaveHoltz as `M`:** 33 stencil steps per application at this grid [P], iteration count grows like `ω` even with Krylov acceleration [S: WaveHoltz20]. No evidence as a preconditioner for `AᴴA`.
- **Sweeping with H-matrix compression, L-sweeps:** these make direct factorisation affordable at scale. In 2D at this size the uncompressed block-tridiagonal form (item 3) is simpler and fits in 69 MB [I].
- **Variable projection and WRI:** ADR-0003. Their preconditioners are still informative (§3).

Kept in reserve (§4): shifted-Laplacian V-cycle on the GPU (for grids beyond direct-solver reach, as in 3D), deflation, truncated Born series, row projections, and full-space Gauss-Newton Krylov (a change of optimiser, needs a maintainer decision).

## 2. Block structure, and the handoff's derivation

### Verified form

Notation at one (shot `s`, frequency `f`): `ũ = u/ρ`, `ρ = u_ref`, `p = |λ_tt|ρ`, `a = w_p/(N_int n_s n_f)`, `b = w_d(ρ/d_s)²/(n_s n_f n_r)`, `κ = dk/dz`, `J = ∂(r/p)/∂z = −diag(∇'²u κ)/p` on interior cells, `P` the receiver sampling, `τ = b|λ_tt|²/a`.

```
H_uu[s,f] = (a/|λ_tt|²) AᴴA + b PᵀP
H_zz      = a Σ_{s,f} JᴴJ                         (diagonal)
H_uz[s,f] = (a/|λ_tt|) Aᴴ J                       (N × N_int)
S         = H_zz − Σ H_zu H_uu⁻¹ H_uz
          = a Σ Jᴴ A⁻ᴴ Pᵀ (τ⁻¹I + Y)⁻¹ P A⁻¹ J,   Y = P A⁻¹ A⁻ᴴ Pᵀ
```

- The first three match autograd Hessian-vector products to 3e-16 on the #96 grid at a warm start [P].
- The closed form of `S` matches a dense Schur complement to 1.2e-12 [P]; its derivation is mine (Woodbury) [I].
- `H_uu` at the true model reproduces #96's row: `λ_min` 3.86e-5, `λ_max` 76.3, κ 1.98e6 [P, R].
- `median(H_uu)/median(H_zz)` comes out at 1.67e4 on the same grid, close to #33's 1.5e4 on tiny [P, R].

Consequences [I, each verified on the small grid in P]:

- `CᴴC = I − H_zz⁻¹ᐟ²SH_zz⁻¹ᐟ²` for `C = H_uu⁻¹ᐟ²H_uzH_zz⁻¹ᐟ²`. At `w_d = 0`, all singular values of `C` are 1 and `S = 0` to round-off (μ range 7e-12). The pilot's top singular value is 1.000000 at every data weight tried.
- `H_zz/diag(S)` has a median of 315 at 2 mm and 1274 at 1 mm [P], close to the 4× per halving expected if a cell's data sensitivity goes like its area [I]. **So the reduced curvature per cell falls like `h²` while `H_zz` doesn't.** The wavefield block's flat iteration count says nothing about this, and ADR-0003's frozen-`c` test can't see it (§9).
- `rank(S)` is bounded by the real data count, as van Leeuwen & Herrmann note for the reduced Hessian (rank at most `ML`) [S: vLH16]. On the #96 grid: 63 of 400 cells with 8 transducers (112 real data), and 15 of 400 with 4 (24 real data) [P]. **At the default problem `rank(S) ≤ 8064` against 15625 model cells**, and likely near 4000 once reciprocity is counted [I]. So at one frequency about half or more of model space has zero Gauss-Newton curvature, and no preconditioner can act on it.
- `S` is the reduced Hessian of the penalty method. As `τ‖Y‖ → 0` it tends to `aτ GᴴG`, the FWI Gauss-Newton Hessian with `G = PA⁻¹J`. As `τ‖Y‖ → ∞` it tends to `a Gᴴ Y⁻¹ G`. Default problem: `τ‖Y‖ ≈ 4700`, `λ̃ = 2.1e-4` at both 2 mm and 1 mm [P]. That's the second limit, so FWI illumination preconditioners target a different operator from `S` [I]: `corr(log diag S, log diag GᴴG) = 0.77` while `corr(log H_zz, log diag GᴴG) = 0.98` [P]. Tape agreement: from outer 20 on, the data term sits orders of magnitude under the PDE term in both arms of record 0002 [R]. Moving `λ̃` toward 1 would mean changing `w_p/w_d`, which belongs to ADR-0002 and #80, not to preconditioning; it would also change what `S` is, so it should precede tuning any model preconditioner [I].

### Discrepancies in the handoff

None changes its conclusions, but four change details:

1. **"Bilinear in `r`" is misworded.** `r` is bilinear in `(u, k)`. The extra coupling is **linear** in `r`: `∂²Φ/∂ũ∂z` gains `−(a/|λ_tt|) Lᴴ diag(κ r/p)`.
2. **A second non-Gauss-Newton term is missing.** `H_zz` also gains `a Re(r̄ (−∇'²u κ'))/p²` from the sigmoid's second derivative `κ' = d²k/dz²`. It stays diagonal, but `H_zz` isn't purely `(dk/dz)²Σ|∇'²u|²` away from `r = 0`. At a randomly perturbed point the Gauss-Newton form alone misses the model row by 49% and the wavefield row by 3.6e-4. Both second-order terms bring them to 5e-16 [P]. Working in `k` removes this term.
3. **The unknowns line is loose.** "`(u_re, u_im, z_m)` per (shot, frequency)" reads as if the latent were per pair. There is one latent; `u` is per pair. The formulas also omit `a`, `b` and `|λ_tt|`; these matter at the default problem because `τ ≈ 6e8`.
4. **#33's "85 on default" was at 60 kHz.** The default band is now 80 kHz [R: #33]. Not re-measured at 80 kHz on the wavefield-curvature median; the analogous figure in `v` is `median(H_zz)/a = 3.2` [P].

Two more things the handoff doesn't state:

- `A = D − diag(k)L` has `k` on the left, and `L` has mirror rows. So `A` is not complex-symmetric once `k` varies; `k⁻¹A` is nearly so [I, from `assemble_H_sparse`].
- "Any preconditioner needs a factored form" is less restrictive than it sounds. Any **fixed linear** right preconditioner `B ≈ A⁻¹` gives `H_uu⁻¹ ≈ BBᴴ`, so `M = B` fits the mechanism. Only `Bᴴ` must be available, and autograd supplies it for torch ops [I]. The SPD-only category is therefore rare.

## 3. Q1: what do full-space and penalty methods use on the coupled Hessian?

**KKT systems (keep `u`, plus multipliers).**

- LNKS solves the KKT system with a Krylov method, preconditioned by a reduced-space factorisation in which forward solves may be replaced by forward preconditioners. Variant `P̃2` keeps two approximate forward solves and an approximate reduced Hessian, and discards second-order terms. It was about 13× and 30× faster than quasi-Newton reduced SQP on Stokes control problems; `P4` and `P2` took about the same number of iterations [S: BirosGhattas05]. Not a wave problem.
- Block-diagonal and constraint preconditioners with multigrid give mesh-independent MINRES and PPCG on Poisson control [S: Rees10]. No indefinite operator, so the multigrid part doesn't carry over; the block recipe does.
- Haber & Ascher use a similar KKT preconditioner for inverse problems, solving the reduced Hessian approximately [S: BirosGhattas05's account; HaberAscher01 (abstract)].
- **The only Helmholtz FWI study I found** is Malovichko et al. They permute the KKT matrix, block-factorise it, then set `F ≈ 0` so `H ≈ L` (a regulariser block). This costs three solves per iteration. With exact `A`, GMRES matched CG on the reduced system. With ILU(21) in place of `A`, time fell from 4800 s to 834 s, about 6× [S: Malovichko22]. Limits: one Gauss-Newton step, part of Marmousi, 120k unknowns per source, 25 sources, serial, CPU. They name shifted-Laplacian multigrid as untested.

**Penalty form (keep `u`, no multipliers): this repo's form.** SLIM's slides give the Newton system and approximate it block-diagonally: `P*P + λ²H*H` on the wavefield, `λ²G_m*G_m` on the model. They report similar quality to WRI with a diagonal Hessian on toy examples and call for more realistic tests [S: Peters14 (slides)]. This is the structure `A(c₀)⁻¹` + `pde_diagonal` approximates, without its data part. van Leeuwen & Herrmann call `λGᵀG` the penalty Hessian's "natural sparse approximation" [S: vLH16], which is exactly `H_zz` here. Result 1 shows when it fails: it needs `μ ≈ 1`, and here `μ ≪ 1` [P].

**ODIL.** Karnakov et al. minimise over fields and parameters together with gradient methods (Adam, L-BFGS-B) or Gauss-Newton with a direct or algebraic multigrid linear solve. For gradient methods they rewrite the fine field as an over-parameterised sum over coarser grids [S: Karnakov24, Karnakov22]. This is a fixed linear change of variables, so it fits `M`; I'd expect it to behave like additive multilevel for the smooth part and do little for near-resonant modes [I]. They don't discuss a preconditioner as such (I read 100k of the paper's 110k characters).

**Contrast source inversion: the closest relative of `initial_model`.** CSI keeps contrast sources as unknowns and uses a fixed background Green's operator, in a two-term objective [S: LinUSE15, vdBK97 (abstract)]. The finite-difference variant factorises the background once and reuses it across iterations and sources [S: FDCSI08 (abstract and secondary accounts)]. Setting `u = u₀ + A₀⁻¹w` turns our residual into `r = w − diag(k−k₀)∇'²u`, CSI's object equation [I]. So `initial_model` is close to FD-CSI in joint form, and CSI's handling of strong contrast deserves a read; I couldn't open the full texts (see Unverified).

**Which keep `u`?** All of the above. WRI and variable projection don't, and need `AᴴA + λPᵀP` factored per model evaluation [S: vLH16; R: ADR-0003]. Their data-space forms, such as the dense `I + μ⁻¹PA⁻¹A⁻ᵀPᵀ` [S: Gholami23], are the origin of candidate 2, which applies the same algebra with fixed `B`.

**How the KKT recipes carry over.** Block-triangular KKT preconditioners are nonsymmetric, so they need GMRES [S: Malovichko22], which doesn't fit L-BFGS. A **congruence** does fit: Biros & Ghattas note congruence-based KKT preconditioning is indefinite and less favourable [S: BirosGhattas05], but here `H` is SPD, so congruence is harmless [I]. The exact one is

```
M = [ B   −H_uu⁻¹ H_uz s ]        MᴴHM = diag(BᴴH_uuB, sᵀSs)
    [ 0    s             ]
```

I call this the **shear**. In `ũ` the PDE-only version is `ũ = |λ_tt| A₀⁻¹ (v + diag(∇'²u₀ κ₀/p) s y)`: a model move carries its first-order wavefield change along. It adds **no extra solve**, since the term joins `v` before the existing solve. A smooth nonlinear variant replaces `κ₀ s y` by `k(z) − k₀`; it's still a fixed reparameterisation, so it doesn't invalidate history [I]. No source tests this in a penalty or ODIL setting. This is my proposal.

### Pilot: spectra of `MᴴHM/a`

`κ⁺` over nonzero eigenvalues, on the #96 grid with 4 transducers: 6672 unknowns, of which 385 are exactly null, from `null(S)` [P].

| Hessian evaluated at | `M` built at | Block-diagonal (now) | + Woodbury | PDE shear, `H_zz` scale | Exact shear, `diag(S)` scale | Exact shear + Woodbury |
|---|---|---|---|---|---|---|
| `c₀` | `c₀` | 1.79e4 | 1.09e3 | 1.20e4 | 156 | 156 |
| overdensity +120 m/s | `c₀` | 1.85e4 | 1.30e3 | 1.24e4 | 171 | 171 |
| `c₀`, ring data | `c₀` | 3.79e5 | 933 | 2.46e5 | 1.41e3 | 138 |
| ring 2000 m/s | `c₀` | 4.09e5 | 4.77e3 | 3.00e5 | 2.38e4 | 8.54e3 |
| ring 3000 m/s | `c₀` | 5.09e7 | 5.42e6 | 1.24e8 | 2.38e8 | 2.37e8 |
| ring 3000 m/s | ring 2000 m/s | 1.09e8 | 2.33e6 | 2.70e8 | 3.89e8 | 3.89e8 |

"Ring" is a 60-cell, 4 mm annulus in 1500 m/s, with data generated on the ring at 3000 m/s and `c` bounds 1300–3100, so compare within a row; the first two rows use the test config's bounds and overdensity data.

Reading [I]:

- **Woodbury is the robust gain:** 9× to 400× at every state, never worse, and it removes all outliers above 10 when fresh. This is why it sits high in the ranking.
- **The shear is excellent near its reference and fragile away from it.** Fresh or at 8% contrast it gains about 100×, and lifts the smallest nonzero eigenvalue from 1.8e-3 to 0.57. At ring 2000 m/s it loses to block-diagonal plus Woodbury (8.5e3 against 4.8e3), and at 3000 m/s it is 44× worse. Refreshing at 2000 m/s doesn't rescue the step to 3000 m/s.
- **`diag(S)` needs the exact shear.** With only the PDE part, the data coupling left behind costs κ about 6.8e3.
- **At high contrast the small end is `S` itself** (1e-5), which nothing here lifts.
- Limits: 1.3 wavelengths across, 4 transducers, a large single jump in `c`. It says nothing about L-BFGS paths, and a real solve moves `c` gradually.

## 4. Q2: the wavefield block

### What the regime actually is

- **Near-resonant regime is a property of the physics, and is larger than #96 saw.** #96's grid has 13 modes below `λ = 1e-3`; Weyl's law `k²Area/4π` predicts 12 [R, I]. The default grid predicts 1526 (568 in the interior). Measured: `σ_min(A)/|λ_tt|` = 0.022 at `c₀` and 0.016 at the true model, against 0.128 on #96's grid; 95 modes below 0.05 at dx = 2 mm and 93 at 1 mm [P]. So the count doesn't grow with resolution, which agrees with #96's physical reading.
- **Skull geometry can trap.** Graham et al. note a sign of the jump gives trapping and resonances super-algebraically close to the real axis for smooth convex interfaces [S: GPS19]. A fast shell around a slow brain has that sign [I]. At 80 kHz the shell (4 mm) is thin against the wavelength in bone (37.5 mm), so strong trapping is doubtful; the 0.022 to 0.016 change is modest [P]. I found no solver study on this geometry apart from [Stanziola25], which is forward-only.
- **PML.** The repo's layer is a graded damping term, not coordinate stretching (`Sponge`, #37). Almost all literature uses a stretched PML or impedance conditions. The evidence below is therefore indirect [I]. Here `σ_max/ω = 2.1` at default, against 18 on #96's grid [P], so PML outliers are much weaker on the default problem, as #96 predicted.
- **Normal equations.** L-BFGS on `H_v = (AB)ᴴ(AB)` sees **singular values** of `AB`; CG on normal equations is governed by singular values and GMRES by eigenvalues [S: NRT92 (abstract)]. Almost all Helmholtz results report GMRES or Bi-CGSTAB, so their iteration counts don't transfer. Sound advice for GMRES (clustered eigenvalues) needn't mean small `κ(AB)²`.

### Measured `κ(H_v)` at default

`κ(H_v) = (σ_max/σ_min)²` of `A·B`, PDE term only, from Lanczos on sparse factors [P]. Rows use exact inverses, so any iterative or approximate variant can only match or lose.

| `B` | at `c₀` | at record 0002's final model | at true model, 2 mm | at true model, 1 mm |
|---|---|---|---|---|
| none (`A`, scaled) | 8.8e5 | | 1.5e7 | 4.1e8 |
| Jacobi | 5.7e5 | 6.1e6 | 1.5e7 | 4.1e8 |
| `A(c₀)⁻¹` | 1 | 1.5e4 | 1.16e5 | 1.57e5 |
| shifted `β = 0.05` | 9.3 | 4.9e3 | 3.6e4 | 5.1e4 |
| shifted `β = 0.1` | 28 | 3.4e3 | 2.2e4 | 3.1e4 |
| shifted `β = 0.2` | 95 | 4.4e3 | 1.9e4 | 2.8e4 |
| shifted `β = 0.5` | 545 | 1.3e4 | 3.8e4 | 5.2e4 |
| shifted `β = 1` | 2.1e3 | 4.3e4 | 9.6e4 | 1.3e5 |
| separable (no `σ_xσ_y`) | 4.1 | 1.5e4 | 1.16e5 | 1.57e5 |
| separable shifted `β = 0.1` | 83 | 3.4e3 | 2.2e4 | 3.1e4 |
| DCT shifted `β = 0.5`, no PML | 5.2e4 | 1.1e5 | 1.09e5 | 1.3e5 |
| `A(c_final)⁻¹` (refreshed) | 1.5e4 | 1 | 1.10e4 | |

Readings [I]:

- **`κ` isn't iterations.** Outliers matter more than the bare ratio. `A(c)A(c₀)⁻¹ − I` has rank at most the number of cells where `k ≠ k₀` (5744 at the true model, 14% of nodes [P]), so `H_v` is identity plus a perturbation of rank at most 11488 per shot. L-BFGS with history 20 can't exploit that the way CG could.
- **A small shift helps when `c` has moved.** `A(c₀)⁻¹` has large gain on `c₀`'s near-resonant modes, which no longer match once `c` moves. The shift caps that gain, and costs at `c₀`. The best `β` is 0.1–0.2 here.
- **Sign warning.** The shift must match the damping, `+iβ|λ_tt|` here. With the opposite sign (`iβλ_tt`, since `λ_tt < 0`), κ reached 5.6e12 in my first run.
- **Nearly flat in `h`.** `A(c₀)⁻¹` and the shifted rows grow about 1.35–1.5× between 2 and 1 mm at the true model, while the unpreconditioned block grows 27×. That's what ADR-0003's two-resolution test wants, though the iteration count itself is untested.

### Method by method

**Shifted Laplacian (CSL)** [S].

- Erlangga et al. use `(β₁,β₂) = (1, 0.5)` with one F(1,1) multigrid cycle inside Bi-CGSTAB. Wedge (1500–2000 m/s): 19 to 37 iterations for 10–30 Hz. Marmousi part: 38, 47, 104, 136 undamped for 1, 10, 20, 30 Hz, at at least 17 points per wavelength; preconditioner and boundary conditions take the true `k(x)` [Erlangga06].
- Theory brackets the shift: wavenumber-independent GMRES needs `ε ≲ k` with exact inversion [GGS15], and multigrid needs `ε ≳ k²` [CocquetGander17 (abstract)]. In between, the SL V-cycle grows fast: 113, 247, 587, 1984 FGMRES iterations for 128² to 1024² on OpenFWI [Lerer24]. With `β = 0.5` (`ε ≈ 0.5k²`) we would be on the multigrid side [I].
- GPU: about 30× over CPU in 2D, single precision [Knibbe11 (abstract)]; more than 25× in 3D [Knibbe13 (abstract)]. No GPU library has it ready; a torch V-cycle (conv2d, transfer via conv) is simple, and autograd provides the adjoint [I].
- Fit: factored if the cycle is fixed and linear. At this size the V-cycle's flop count is only about 4× below sparse LU's (cost table below), so there's no gain until the grid outgrows a direct solver [I].
- **Evidence gap:** nothing on singular values, on the skull, or with a damping layer.

**Separation of variables** [S]. Plessix & Mulder's preconditioner works for smooth models and low frequencies but "rapidly deteriorates" with roughness or frequency [PlessixMulder03 (abstract)]. On Marmousi it took 648 iterations at 20 Hz and failed by 2000 at 30 Hz, against the CSL's far better scaling [Erlangga04]. Here it's an exact `A(c₀)⁻¹` up to the corner term `σ_xσ_y`, not an approximation to a rough model. It inherits `A(c₀)⁻¹`'s staleness exactly (same κ to three digits at the true model [P]). Eigenvectors have `cond(V) = 2.8e4`, float32 round trip 4e-4 [P]: usable, worth watching. Only valid for a homogeneous reference, so only the first band.

**Sweeping** [S]. GMRES in under 5 iterations for three velocity fields with PML, essentially independent of size, in 2D with near-linear cost [EngquistYing11]. L-sweeps parallelise this [Taus20]. Fit: factored, with a per-band layer factorisation. Sweeping-type methods are approximate block LU factorisations [GanderZhang19 (abstract)], and the exact version in 2D is the block-tridiagonal LDU of item 3. Flat evidence, with no GPU library that I found, and no evidence on singular values.

**Domain decomposition** [S]. With absorption `|ε| ∼ k²`, overlapping additive Schwarz performs optimally for the absorptive problem, in theory and in the authors' tests [GSV17]. On Marmousi at 5 points per wavelength, two-level coarse-grid methods need 15–44 GMRES iterations up to 160 subdomains and 20 Hz, while one-level grows to 373 [Bootland20]. GenEO theory extends to indefinite problems [Bootland21 (abstract)]. On a single GPU in 2D, 16 subdomains' local factors have about the same `nnz` as the global LU, so nothing is gained [I].

**WaveHoltz** [S]. Turns Helmholtz into a positive definite system via time-domain filtering; Krylov-accelerated iterations scale like `ω`, with deterioration only in a narrow band around a resonance in a 1D Dirichlet test; CG applies in the energy-conserving case and GMRES with impedance boundaries [WaveHoltz20]. No PML-type damping layer in the parts I read. Here a fixed number of steps is a linear `B` that fits, and each step runs on the GPU. 33 steps per period [P]; no evidence that a single application is a useful `B`.

**Row projection, CARP-CG** [S]. The one family built for positive semi-definite solves: Kaczmarz double sweeps turn any system into SPD `I − Q`. Better clustered than normal equations, or Jacobi-scaled normal equations, in 1D. 3D salt model: 180, 267, 443 iterations at `ε = 1e-3` as frequency doubles, with `(4k+2)N` flops per apply [vanLeeuwen12]. Reported to beat a shifted-Laplacian multigrid [GordonGordon13 (abstract)]. Kaczmarz is sequential; CARP parallelises with block averaging. On a GPU, fixed multi-colouring is my suggestion, with no source. Fit: linear and fixed if sweep count and colouring are fixed, and SSOR-like factors are available [I]. Reserve.

**Deflation** [S]. Two-level deflation reaches near-constant iterations in 1D and 2D with higher-order vectors, for example 9 iterations at `k = 1000` with an approximately inverted CSL, and is robust on Marmousi [DwarkaVuik20]. Earlier analysis notes more near-zero eigenvalues in 2D than in 1D [Sheikh13]. Eigenvector deflation here would need at least 100–300 vectors, per frequency but shared by shots. Target modes **move with `c`** [I], so it goes stale on exactly what it targets, and `A(c₀)⁻¹` already handles these at `c₀`.

**Convergent Born** [S]. `ε ≥ max|k(r)² − k₀²|` guarantees convergence, with pseudo-propagation of `2k₀/ε` per iteration, so higher contrast costs more [CBS16]. The split preconditioner generalises it [Vettenburg23]. Stanziola et al. apply this to a skull at 2800 m/s (3D, with density), with adjoints, and on the GPU; they report contrast raising time to converge, and a trade-off in absorbing-layer design [Stanziola25]. Arithmetic: `k` from 168 to 347 rad/m for 3000–1450 m/s, so `ε = 4.6e4`, `k₀ = 272`, 11.8 mm per iteration, and 35 iterations to cross 410 mm once [I]. Fixed `K` gives a linear map; `B` may depend on `c(z)` while still being a fixed function of the unknowns, which is the only candidate that tracks `c` with no factorisation (§6). `wavesim` is a torch implementation on CUDA [S: wavesim]. Its `(L+I)⁻¹` uses an FFT Laplacian rather than this repo's 2nd-order stencil, so swap in the FD symbol for an inverse-crime match [I]. Reserve, if refresh and shift prove insufficient.

**Approximate inverses and incomplete factorisation** [S]. See ruled-out list. The inverse of `A` is a Green's function and non-local [I].

**Learned** [S]. U-Net as coarse solver or deflation with SL V-cycles inside FGMRES, plus mini-retraining [Azulay22]. Encoder-solver trained up to 512²: 143 iterations at 4096² on OpenFWI against more than 2000 for the V-cycle [Lerer24]. GPU natural; nonlinear ("non-linear coarse grid correction"). Stanziola's learned solver [Stanziola21] and learned Born series [LBS22] target the transcranial setting, as solvers. Fit: only if restricted to a linear network; no evidence for that.

### Costs at the default problem

`N = 205² = 42025`, 64 shots, complex64. `u` is `64 × 42025 × 8 B = 21.5 MB`; L-BFGS history 20 is `2 × 20 × 21.6 MB = 864 MB`. Unpreconditioned `A` apply (5-point): about 30 flop per node, so `30 × 42025 × 64 = 8.1e7` flop; `A + Aᴴ` per evaluation = 1.6e8. Measured unpreconditioned evaluation 0.145 s [R].

Per evaluation means one `M` and one `Mᴴ` over all shots at one frequency. Complex multiply = 8 flop.

| Candidate | Memory per frequency | Flop per evaluation | Arithmetic and notes |
|---|---|---|---|
| splu `A(c₀)⁻¹` | 30 MB | 3.8e9 | `nnz(L+U) = 3.71e6` [P], `× 8 × 64 × 2`. Measured 0.348 s/eval (737 s / 2118) on CPU [R]. 1 mm: `nnz = 1.89e7`, 151 MB [P]. |
| Same, cuDSS | as above | 3.8e9 | Complex, `A`, `Aᵀ`, `Aᴴ`, multi-RHS, hybrid memory, Linux and Windows CUDA [S: cuDSS]. Not MPS. |
| `torch.sparse.spsolve` | | | Doesn't suit: docs show no factor object, only `left=True` [S: torchspsolve]; reuse across calls appears impossible [I]. |
| Block-tridiagonal LDU (dense `D⁻¹`) | `205 × 205² × 8 B` = 69 MB | 1.8e10 | `2 × 205 × 205²` cmul per RHS `× 8 × 64 × 2`. Setup about `205 × 2 × 205³ × 8 = 2.8e10`. Heterogeneous `c` fine. Complex `linalg.inv` and `lu_factor` fail on MPS; matmul works [P], so set up on CPU, apply on MPS. |
| Separable fast diagonalisation | `4 × 205² × 8 B` = 1.3 MB | 3.5e10 | `4 × 205 × 205²` cmul per RHS. Homogeneous reference only. Same for separable shifted. |
| DCT shifted inverse | 0.3 MB | 8.9e8 | `5N log₂N = 3.2e6` per FFT, two plus a scale, `× 64 × 2`. `torch.fft` runs on MPS [P]. No PML. |
| CSL V-cycle | about 5 MB | 8.6e8 | About 4 stencil-equivalents per level, `× 4/3`, `× 30N × 64 × 2`. F-cycle about ×1.5. |
| Data Woodbury | 21.5 MB | 5.5e9 | Shared basis `64 × N × 8 B`; `2 × 64 × N` cmul per RHS `× 8 × 64 × 2`. One 64-RHS adjoint solve (0.27 s) plus a 64×64 eigenproblem, at set-up. |
| Shear, PDE part | 8 MB | about 8e6 | One diagonal per shot on the interior; no added solve. |
| Shear, data correction | as Woodbury | 5.5e9 | Low-rank term. |
| `diag(S)` scale | one field | 0 | Set-up only: one 64-RHS adjoint solve. |
| Truncated Born, `K = 35` | about 0.3 MB | 3.1e10 | `35 × 6.9e6 × 64 × 2`. |
| Eigenvector deflation, 300 modes | 100 MB | 2.6e10 | `4 × 300 × N × 64` cmul. Eigenvectors are expensive at set-up. |
| CARP double sweep, 5-point | small | 9.5e8 | `22N` complex ops `× 8 × 64 × 2`; needs colouring. |
| WaveHoltz, one iteration | small | 5.3e9 | `33 × 30N × 64 × 2`. |
| Sparse approximate inverse, 25 nonzeros per row | 8.4 MB | 1.1e9 | `25N × 8 × 64 × 2`. Quality is the issue, not cost. |
| Learned | not estimated | not estimated | Depends on the network; I didn't extract an operation count. |

Conclusion from the table [I]: 2D fill-in is mild. Any GPU candidate wins by parallelism, not flops. On MPS, complex dense matmul and FFT work today; complex LU doesn't [P].

## 5. Q3: the model block and coupling

### What beats a diagonal?

Literature on reduced FWI Hessians, all targeting `GᴴG` [S]:

- **Pseudo-Hessian** estimates the diagonal from forward wavefields [Shin01 (abstract)]. Our `H_zz` is this exact object in penalty form [I]: `J` is Shin's virtual source.
- **Plessix-Mulder weights** derive from a Born Hessian without assuming infinite receiver coverage [PlessixMulder04 (abstract)].
- **Diagonal GN from reciprocity:** Pan et al. build a pseudo-diagonal Gauss-Newton Hessian from reciprocity and report that an l-BFGS preconditioner seeded with it works best in Hessian-free GN [Pan15].
- **Matrix probing** fits a Hessian-inverse symbol from a few Hessian applications. On Marmousi-derived tests, randomized trial functions stay robust, while Krylov-type ones degrade with order; the authors warn about complicated kinematics such as shadow zones [Demanet12].
- **Model smoothing and reparameterisation** add structure via a preconditioning change of variables [Guitton12 (abstract)]. Fits `M`, GPU via conv, targets #111's λ/2 fringes. A coarser model grid or multilevel latent would also remove the `h²` drift above, and shrink `null(S)`: 2 mm cells are far below the λ/2 = 9.4 mm resolution of one 80 kHz band. Trades resolution, which matters on a 4 mm skull rim [I].

**For the joint form, the right target is `S`** [I]:

- `diag(S)` needs no new factorisation: with `Z = A⁻ᴴPᵀ` from one 64-RHS adjoint solve, `S_ii = a Σ_s |J_ii|² (g_i)ᴴ K_s g_i`, with `g_i = conj(Z[i,:])` and `K_s = (τ⁻¹I + Y)⁻¹`. This was run on the default problem in seconds [P].
- Its shape agrees with `H_zz` to about 3×: `H_zz/diag(S)` has quantiles 240 / 315 / 753 at 1 / 50 / 99% at 2 mm, and 976 / 1274 / 3196 at 1 mm [P]. The level differs by 315× to 1274×. So `pde_diagonal`'s shape is roughly right but its **level isn't the Schur level**, and that level can't be used without a shear because `H_zz` then becomes a 315× outlier. That's why the shear and `diag(S)` go together, and why this pair inherits the shear's fragility at contrast (§3).
- Full low-rank `S` is feasible in 2D [I]. `G` rows are products of two stored fields; with 8064 real data, a dense data-space factorisation is about 1e12 flop, 1 GB. This is a band-fixed reduced Gauss-Newton Hessian, kept as a block on the model side. Same staleness risk as the shear.

### The `pde_diagonal` plateau and drift of #98

No source addresses this. Hypotheses, with tests [I]:

- **Drift (case 2).** At `r ≠ 0`, the model gradient `Jᴴr` and the wavefield gradient have the same power to reduce the PDE term (coupling singular values of 1). With `pde_diagonal`, the two have the same curvature, so `c` absorbs warm-start error as readily as `u`. Without it `c`'s curvature is 85× to 1.5e4× lower [R] and drift is under 0.14% [R]. Return is set by `μ ≪ 1`, which would be slow, and #98 didn't look past 50 iterations.
  - *Test:* run case 2 longer, and with the shear. At first order, the PDE term no longer pulls `y`, so drift should drop sharply if this is right.
- **Plateau (case 3).** Frozen `u` recovers `c` to 2.6e-6 [R], since a known `u` supplies a pointwise equation at each cell. In the joint solve, once `r` is small, the remaining motion is along the valley, at the `S` scale. On the unit problem `rank(S) = 63 of 400` [P], so **there is a large family of `(c, u)` with objective near zero**. The 0.39% may be partly convergence to another near-minimiser, not a stall.
  - *Test:* at the plateau iterate, split `c − c_true` into `range(S)` and `null(S)` with a dense SVD. If it's mostly `null(S)`, no preconditioner helps; the fix is data (frequencies, acquisition) or a regulariser. Sweeping `eps_precond` tests #98's first suspect; #33 found no floored cells on the warm start, so I expect little.
- **Stale scale.** `s` is set at warm start. #98's case 2 wavefield is wrong and `κ(z)` changes as `z` moves (skull cells reach 0.07 of the maximum slope [R: #111]). The `κ` part is exactly known, so a raw `k` latent removes that part of staleness and the sigmoid's second-order term (#35; §2) [I].

### Change of parameterisation

`r` is linear in `k`, so `H_kk` has no `r` term, and `H_kk`'s level follows `|∇'²u|²` only [R: #33]. Sigmoid-bounded `k` is an affine relabelling of sigmoid-bounded `m` [R: #18], so only a projected raw `k` (#35) changes anything. Not itself a preconditioner, but it would make any model preconditioner less stale.

## 6. Q4: a preconditioner that goes stale

**Three distinct things are called "refresh".**

1. **Restart with new `M`.** What ADR-0003 and #104 rule out inside a band. Costs factors (cheap here) and loses the 20 pairs.
2. **`z`-dependent `M`, as a fixed function.** `u = B(c(z)) v` isn't a refresh: it's a smooth fixed reparameterisation of the joint space, so L-BFGS history stays valid [I]. With exact `A(c)⁻¹` it would be source extension; with truncated Born it needs no factorisation, and keeps `v` as an unknown. It changes coupling, and the mapping's derivative enters the gradient, so this is research, not a drop-in. Interacts with ADR-0003's spirit even if not its letter (flag for review).
3. **Remap the history.** See below.

**L-BFGS history isn't inherently invalidated by a new `M`** [I]. Curvature pairs are secant pairs of the same function, so a linear change of coordinates maps them exactly: `s_v' = M'⁻¹M s_v`, `y_v' = M'ᴴM⁻ᴴ y_v`, with inner products preserved. With `M = |λ|A⁻¹`, each pair costs one forward solve with the old factors and one adjoint solve with the new, as `M⁻¹` and `M⁻ᴴ` are stencils. 20 pairs ≈ 20 × (0.16 + 0.27) s ≈ 8.6 s on CPU float64 [P], about one outer iteration. `torch.optim.LBFGS` keeps these in `old_dirs`/`old_stps`/`prev_flat_grad`/`H_diag` state [S: torchLBFGS], which is editable. What's lost is the accuracy of old pairs when the Hessian has moved, not validity. #33/#104's statement holds for naive restarts, not in principle. **Any scheme here must state that it requires explicit history handling, as the handoff asks.** Pairs measured before a big `c` move may be misleading regardless.

**`H0` hook against change of variables** [I]. L-BFGS in `v` with `γI` is the same as L-BFGS in `u` with `H0 = γMMᴴ`, with `γ = sᴴy / yᴴMMᴴy`: the two-loop recursion transforms term by term. Differences:

| | Change of variables (now) | Custom `H0` |
|---|---|---|
| Applies | `M + Mᴴ` per evaluation, so 2118 evals in record 0002 | `MMᴴ` per iteration, about 2000 (saves about 5.6%) |
| Line search | Must run `M` | Needs no preconditioner |
| Needs | Factored `M` | Any SPD `H0`; no square root |
| Refresh | Needs history mapping | Just use a new `H0`: pairs are physical, and `H0` may change each iteration [S: LiuNocedal89 §4 "it is preferable to change this matrix as we proceed"] |
| Per-block `γ` | Not possible | Possible; `sᴴy` can be negative per block, so it needs a guard. No source tests this |
| Cost to build | Existing | About 100 lines; loses stock strong-Wolfe wrapper |

**Literature evidence** [S]:

- Variable `H0` is standard: scaling by `γ_k` per iteration, and diagonal alternatives [LiuNocedal89].
- Liu et al. compare diagonal `H0` choices in BFGS for elastic FWI and note performance may depend on the initial guess [LiuBFGS20].
- Flexible CG: convergence is roughly unaffected for small variations in the preconditioner; a flexible variant can beat standard CG [Notay00 (abstract)].
- L-BFGS as a preconditioner for CG, in Hessian-free Newton [MoralesNocedal00 (abstract)], [Pan15]; Métivier notes it is weak in the first iterations, since it must first accumulate pairs [MetivierHDR17].
- Multi-preconditioned L-BFGS uses a nonlinear change of coordinates and separate local and global memories, in FBPINNs only [MP-LBFGS26].
- **Hessian-free or Gauss-Newton:** Malovichko's 6× is the only cost evidence on Helmholtz [Malovichko22]. Newton takes about 10 iterations against about 10000 for L-BFGS-B on an ODIL flow problem [Karnakov22]. PINN work links differential-operator conditioning to loss conditioning, and compares Adam, L-BFGS and a Nyström Newton-CG on it [Rathore24]. On #96's quadratic, CG beats L-BFGS by 6.4× on the unit grid and by 1.3× at 77² [R], which caps what history gains for frozen `c`.
- **No source I found quantifies restart against reuse cost** for FWI or ODIL.

**Cost evidence here.** Factor: 0.14–0.38 s, 64-RHS solves 0.16 + 0.27 s, on CPU float64 [P]. 1 mm: 1.5–2.8 s, and 1.1 + 1.8 s [P]. An outer step of record 0002 is 10.3 s [R]. So a refresh every outer step would add about 3–4%, excluding lost history and warm-start effects.

**Band splitting needs no code** [I]: repeat the same frequency in `continuation.bands` so each band re-factorises at its initial model. It's ADR-compliant as written. Caveat: the warm start also resets `u` to `A(c)⁻¹q`, so `r = 0` and `∇_zΦ = 0` restart each time (#18 §3.3). A shift also breaks "`T` uniform" (`preconditioned_wavefield_curvature`), so `pde_diagonal` needs a new `T`.

## 7. Candidate table

Form: **F** factored `M` (fits the change of variables). **H** SPD-only `H0` (needs custom L-BFGS). **N** nonlinear or variable (fits neither without change). All keep `u` as an unknown. "ADR" flags factorisation use: ✓ fine as worded; **note** complies with "no refactorisation per evaluation" but the ADR's wording ("factorises `A(c)` only for the warm start") would need amending. Refresh: whether it needs one in a band, and what that does to history. Regime column: evidence at indefinite, PML, skull contrast. Costs are in §4. Under ADR-0003's test, every wavefield candidate runs the protocol in §9.

| Candidate | Form | ADR | Refresh in band? | GPU, library | Regime evidence | ADR test notes |
|---|---|---|---|---|---|---|
| `A(c₀)⁻¹` (now) | F | ✓ | No; goes stale, κ 1.16e5 [P] | Solves on CPU; cuDSS possible | #96 flat at 8% [R]; record 0002 at skull [R] | Passed at 8% only; rerun on default true model |
| Shifted factors | F | note | Optional | cuDSS or block LDU | Theory milder [S]; κ table [P] | Sweep `β` |
| Separable, optionally shifted | F | ✓ when unshifted | No; first band only | Torch dense matmul, MPS OK | Exact `A(c₀)⁻¹` up to corners [P] | Must match `A(c₀)⁻¹` counts |
| Block-tridiagonal LDU | F | ✓ | Per band or by history mapping | Torch; set-up on CPU on MPS | Exact, so as `A(c₀)⁻¹` | Same counts, compare wall time |
| Data Woodbury | F | ✓ | Never (no `c` in the data term; depends only on `B`) | Torch matmul | Pilot only: 9–400× in κ⁺ [P] | Vary `n_r`; expect biggest effect at many receivers |
| Shear + `diag(S)` | F | ✓ | Goes stale fast: at high contrast worse than block-diagonal [P] | Torch | Pilot only; see §3 | No frozen-`c` effect by design. Test on #93 cases 2 and 3, then joint |
| CSL V-cycle | F if fixed cycle | ✓ | Reference `c` can update cheaply, but history is the issue | None packaged; torch conv | Marmousi, wedge, GMRES or Bi-CGSTAB only [S]; shift gap [S] | Compare with exact shifted row to see the V-cycle's loss |
| DCT shifted, no PML | F | ✓ | No | `torch.fft` | κ poor at `c₀` [P] | Control only |
| Sweeping, L-sweeps | F | note | Per band | No library found | Under 5 GMRES iterations, 2D with PML [S]; no skull | Unnecessary in 2D [I] |
| Additive Schwarz, two-level | F | note | Per band | Partial via local LU | Marmousi 15–44 GMRES iterations [S] | Low priority |
| WaveHoltz, fixed steps | F | ✓ | Reference updates freely | Torch stencil | Scales `~ω`; near resonance worse [S] | Only if a single apply shows value |
| CARP, Kaczmarz fixed sweeps | F | ✓ | Free | Custom colouring; none packaged | 3D salt; scales with `f` [S]; no skull | Reserve |
| Deflation | F (projector) | note | Modes move with `c` | Torch matmul | Near-constant iterations 2D, Marmousi [S]; GMRES only | Reserve |
| Truncated Born, fixed `K` | F; `c(z)`-dependent: fixed function | ✓ | None needed in `c(z)` form | `torch.fft`, `wavesim` (CUDA) | Skull at 2800 m/s, forward-only [S] | Reserve |
| Born or Krylov to tolerance | N | ✓ | n/a | | | Not usable |
| ILU, SPAI, FSAI, AMG | F (if fixed) | note | Per band | AmgX (complex limited), Ginkgo (complex not stated) | Fails or needs high fill [S] | Ruled out |
| Learned U-Net | N (unless linear) | ✓ | Retrain | GPU natural | OpenFWI, CIFAR; FGMRES [S]; no skull preconditioning | Ruled out now |
| Jacobi | F | ✓ | No | Trivial | #96 [R] | Done: fails |
| Pseudo-Hessian, `pde_diagonal` | F | ✓ | No, per band | Trivial | Reduced FWI [S]; #98 [R] | Model block: §5 |
| `diag(S)` alone | F | ✓ | Per band | Trivial | Pilot only | Unusable without shear: H_zz becomes an outlier [P] |
| Low-rank dense `S` | F | ✓ | Goes stale | Dense on GPU, 1 GB | None | After measurement |
| Matrix probing, PSF | F | ✓ | Goes stale | Conv | Marmousi-derived [S]; not joint | Target `S` first |
| Model smoothing | F | ✓ | No | Conv | [S] (abstract) | Resolution trade-off |
| Custom L-BFGS `H0` | H | ✓ | Free, pairs stay valid | Anything | Variable `H0` is standard [S] | Equivalent to `M` at fixed `H0`: check |
| Full-space GN Krylov | not L-BFGS | ✓ | Natural per GN step | Any | Marmousi part, serial, 6× [S] | Needs a maintainer decision |

## 8. Joint-spectrum measurement

Extends #96's method: same real-pair Hessian, same check against autograd HVP (3e-16). Adds the model block, per-shot wavefield blocks, and coupling. Two parts are already piloted (§2, §3).

### Grids

| Tier | Problem | Unknowns | Method | Time |
|---|---|---|---|---|
| 0, dense | #96's 28² (20² + PML 4), 4–8 transducers, float64 | 6672–12944 | Assemble blocks from `assemble_H_sparse` and `extract_observations`; `eigvalsh` | 15–130 s (piloted) |
| 1, block-dense | 39² + PML 8 and 77² + PML 16 of the same physical problem, plus a coarse default: 62² + PML 20 at 4 mm, 80 kHz, 16 transducers (13.3 wavelengths, 4.7 points per wavelength) | 1e5–1e6 | Dense `S` from the closed form (§2); `1 ± √(1−μ)` gives exact-block spectra with no joint eigensolve | Minutes |
| 2, matrix-free | Default 205², float64, 64 shots | 2.7M + 15625 | Lanczos or LOBPCG with analytic GN matvec for extremes; stochastic Lanczos for density; `S` by closed form | Hours |

Note on autograd: `_FactorSolveFn` has no second derivative, so at tier 2 use the analytic Gauss-Newton matvec in `(v, y)` and check it against a finite difference.

### States

(a) warm start at `c₀`; (b) true model with true wavefield (`r = 0`); (c) record 0002's final model (`arrays.npz` has `c`; `u` only for 4 shots, so use `A(c)⁻¹q` plus a note, or re-run from a snapshot); (d) #93 case 2 and 3 iterates, tier 0, `r ≠ 0`, including second-order terms.

### Report

1. **Per block:** spectrum of `H_uu` per shot and in `v`; `H_zz` quantiles; data outliers `1 + τ eig(Y)` and their count; `λ̃`, per frequency and at two resolutions (#80's question).
2. **Coupling:** singular values of `C`, and `corr(log H_zz, log diag S)`.
3. **Schur:** `μ` spectrum, `rank(S)`, `dim null(S)`, `κ⁺`.
4. **Joint:** raw `H`, and `MᴴHM` under each `M`; `κ⁺`, cluster count (say, number of log-spaced 0.1-decade bins needed to hold 99% of nonzero eigenvalues, or the number of distinct clusters by largest-gap splitting), and the count of outliers outside [0.5, 2].
5. **Where the error lies:** as #96's table, project `(u − u*, z − z*)` onto eigenvectors. Add the `range(S)`/`null(S)` split of the model error.
6. **Staleness:** the above with `M` from `c₀` at states (b) and (c), and with rank of `A(c)A(c₀)⁻¹ − I` against the moved-cell count.

### How it decides

| Observation | Decision |
|---|---|
| Model error mostly in `null(S)` at #98's plateau | No preconditioner helps; go to frequencies, acquisition or a regulariser |
| `λ_max` of the preconditioned block is data outliers, and many | Candidate 2 (Woodbury) first |
| Outliers track moved cells | Shift (1), then refresh (4) |
| `κ⁺` set by small `μ`, with shear lifting it at states (b), (c) | Shear + `diag(S)` (5), then low-rank `S` |
| Shear gain vanishes at (c), as the pilot suggests | Drop shear; use refresh, or `z`-dependent map |
| `μ` spread wide and low-rank fits | Dense `S`, or matrix probing |
| `H_uu` near-resonant modes hold most `u` error under `M` | Shift, deflation, Born |

## 9. How to evaluate under ADR-0003

Protocol, applied to each wavefield candidate:

- **Setup:** model frozen at the true model, warm start on the initial model, L-BFGS on the wavefield block. Same physical problem at two resolutions: the #96 grid pair, **and** default at 4 mm and 2 mm (62² + PML 20, 125² + PML 40), Shepp-Logan true model. #96's 8% overdensity doesn't test the skull (κ 2.2 against 1.16e5 [P]).
- **Report:** iterations, evaluations and wall time to fixed wavefield error (1e-3, plus 1e-2 at default, in case 1e-3 is out of reach), by interior and PML. Also the singular-value row from §4.
- **Pass:** fewer iterations than `A(c₀)⁻¹`, or the same at lower wall time on GPU.
- **Then, in the joint solve:** repeat record 0002's design as an experiment with `log_every = 1` (the record's own fix for its knife-edge cost reading).

**A gap in the test.** It fixes the model, so it can't see the coupling or the `h²` drift in §2. A candidate can pass it and leave the joint solve unchanged: record 0002 did about that, with 0.23× wavefield error and a skull still far off. A two-resolution joint measurement should sit beside it, for instance `μ` and `κ⁺` from §8.

Candidate notes: exact replacements must match `A(c₀)⁻¹` counts within noise. Shear and `diag(S)` act on coupling, so frozen-`c` doesn't test them: use #93 cases 2 and 3 plus joint. Woodbury needs a receiver-count axis. A pass at 8% isn't a pass at skull contrast.

## 10. Unverified and limits

**Not retrieved in full; only abstract or search excerpts:**

- HaberAscher01, CocquetGander17, ErnstGander12, GanderZhang19, NRT92, GordonGordon13, Bootland21, MoralesNocedal00, Notay00, Shin01, PlessixMulder03, PlessixMulder04, Guitton12, vdBK97, FDCSI08, Knibbe11 and Knibbe13 (abstract read, tables not).
- Sources whose full text I couldn't open: `repository.tudelft.nl` (connection refused, FD-CSI and Ozmen), `hal` (SEISCOPE toolbox), `earthdoc` and `sciencegate` (403).

**Named in the brief or in my reading, not verified, not relied on:**

- Shin's pseudo-Hessian body text; FSAI (Kolotilina & Yeremin), SPAI (Grote & Huckle); Anzt et al. ISAI.
- Pearson & Wathen and Murphy, Golub & Wathen (search excerpts only: exact-Schur block preconditioners give 2 or 3 eigenvalues, which is the indefinite analogue of the `{0, 1, 2}` here).
- Akçelik et al. SC02; Bridson & Greif multi-preconditioned CG; Golub & Ye inexact CG; Brune et al. nonlinear preconditioning; Stolk 2013; Ozmen et al. (CSI on breast ultrasound, search excerpt); SEISCOPE preconditioned L-BFGS; Björck & Elfving CGMN (via vanLeeuwen12 only); BPX; stochastic Lanczos quadrature; Fletcher and Valenciano PSF deconvolution; Modrak & Tromp; Tarantola.
- Whether `torch.sparse.spsolve` uses cuDSS or supports complex: docs don't say.

**Limits of the pilot:**

- `κ` is not iteration count; L-BFGS with history 20 is not CG; PDE term only unless stated.
- Default rows use exact factors, float64, on CPU. Float32 behaviour isn't tested, and record 0002's float32 stall (`464763e`) warns that scale matters.
- The small-grid `MᴴHM` table is 1.3 wavelengths across, with 4 transducers and a 60-cell ring. It says little about many-mode behaviour, and nothing about L-BFGS.
- GPU timing isn't measured; flop counts assume nothing about throughput. Scratch scripts were in `/tmp` and aren't committed.

## Sources

Retrieved in this session. Read = targeted sections of the full text.

- [BirosGhattas05] Biros G., Ghattas O. "Parallel Lagrange-Newton-Krylov-Schur methods for PDE-constrained optimization. Part I." *SIAM J. Sci. Comput.* 27(2):687–713. Read. https://users.oden.utexas.edu/~omar/papers/lnks/lnks1.pdf
- [Rees10] Rees T., Dollar H.S., Wathen A.J. "Optimal solvers for PDE-constrained optimization." *SIAM J. Sci. Comput.* 32(1):271–298. Preprint, section headings and structure read. https://numerical.rl.ac.uk/media/people/tyrone-rees/pdf/NA-08-10_new.pdf
- [HaberAscher01] Haber E., Ascher U.M. "Preconditioned all-at-once methods for large, sparse parameter estimation problems." *Inverse Problems* 17:1847–1864. Abstract. https://doi.org/10.1088/0266-5611/17/6/319
- [Malovichko22] Malovichko M., Orazbayev A., Khokhlov N. "Iterative PDE-constrained optimization for seismic full-waveform inversion." arXiv:2204.06489. Read. https://arxiv.org/abs/2204.06489
- [vLH16] van Leeuwen T., Herrmann F.J. "A penalty method for PDE-constrained optimization in inverse problems." *Inverse Problems* 32:015007. arXiv:1504.02249. Read. https://arxiv.org/abs/1504.02249
- [Peters14] Peters B. "Quadratic-penalty based full-space methods for waveform inversion." SINBAD 2014 slides. Read. https://slim.gatech.edu/Publications/Public/Conferences/SINBAD/2014/Fall/peters2014SINBADqpb/peters2014SINBADqpb.pdf
- [Gholami23] Gholami A. "Full waveform inversion and Lagrange multipliers." arXiv:2311.11010. Read. https://arxiv.org/abs/2311.11010
- [Karnakov24] Karnakov P., Litvinov S., Koumoutsakos P. "Solving inverse problems in physics by optimizing a discrete loss." *PNAS Nexus* 3(1):pgae005. Read through a page summary. https://pmc.ncbi.nlm.nih.gov/articles/PMC10799659/
- [Karnakov22] Karnakov P., Litvinov S., Koumoutsakos P. "Optimizing a DIscrete Loss (ODIL)." arXiv:2205.04611. Read. https://arxiv.org/abs/2205.04611
- [vdBK97] van den Berg P.M., Kleinman R.E. "A contrast source inversion method." *Inverse Problems* 13. Abstract. https://doi.org/10.1088/0266-5611/13/6/013
- [FDCSI08] Abubakar A., Hu W., van den Berg P.M., Habashy T.M. "A finite-difference contrast source inversion method." *Inverse Problems* (2008). Abstract and secondary accounts. https://doi.org/10.1088/0266-5611/24/6/065004 (title confirmed through Crossref; text not opened)
- [LinUSE15] Lin et al. "Robust contrast source inversion method for waveform reconstruction of the breast tissue in ultrasound computed tomography." USE 2015. Read. https://senkyo.co.jp/use/proceedings/USE15/pdf/1J1-4.pdf
- [Erlangga06] Erlangga Y.A., Oosterlee C.W., Vuik C. "A novel multigrid based preconditioner for heterogeneous Helmholtz problems." *SIAM J. Sci. Comput.* 27(4):1471–1492. Read. https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Erl06OV.pdf
- [Erlangga04] Erlangga Y.A., Vuik C., Oosterlee C.W., Plessix R.-E., Mulder W.A. "A robust iterative solver for the two-way wave equation based on a complex shifted-Laplace preconditioner." SEG 2004. Read. https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Erl04VOPM.pdf
- [GGS15] Gander M.J., Graham I.G., Spence E.A. "Applying GMRES to the Helmholtz equation with shifted Laplacian preconditioning." *Numer. Math.* (2015). Preprint read. https://people.bath.ac.uk/eas25/GaGrSp14.pdf
- [CocquetGander17] Cocquet P.-H., Gander M.J. "How large a shift is needed in the shifted Helmholtz preconditioner for its effective inversion by multigrid?" *SIAM J. Sci. Comput.* 39(2):A438–A478. Abstract. https://archive-ouverte.unige.ch/unige:168606
- [Knibbe11] Knibbe H., Oosterlee C.W., Vuik C. "GPU implementation of a Helmholtz Krylov solver preconditioned by a shifted Laplace multigrid method." *J. Comput. Appl. Math.* 236:281–293. Abstract. https://ir.cwi.nl/pub/18787/18787D.pdf
- [Knibbe13] Knibbe H., Oosterlee C.W., Vuik C. "3D Helmholtz Krylov solver preconditioned by a shifted Laplace multigrid method on multi-GPUs." Abstract. https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Kni13OV.pdf
- [PlessixMulder03] Plessix R.-E., Mulder W.A. "Separation-of-variables as a preconditioner for an iterative Helmholtz solver." *Appl. Numer. Math.* 44. Abstract. https://doi.org/10.1016/S0168-9274(02)00165-4
- [GanderZhang19] Gander M.J., Zhang H. "A class of iterative solvers for the Helmholtz equation: factorizations, sweeping preconditioners, source transfer, single layer potentials, polarized traces, and optimized Schwarz methods." *SIAM Rev.* 61(1):3–76. Abstract. https://archive-ouverte.unige.ch/unige:168643
- [EngquistYing11] Engquist B., Ying L. "Sweeping preconditioner for the Helmholtz equation: hierarchical matrix representation." *Comm. Pure Appl. Math.* 64. Read. https://arxiv.org/abs/1007.4290
- [Taus20] Taus M., Zepeda-Núñez L., Hewett R.J., Demanet L. "L-sweeps." *J. Comput. Phys.* 420:109706. Read. https://arxiv.org/abs/1909.01467
- [GSV17] Graham I.G., Spence E.A., Vainikko E. "Domain decomposition preconditioning for high-frequency Helmholtz problems with absorption." *Math. Comp.* Read. https://arxiv.org/abs/1507.02097
- [Bootland20] Bootland N., Dolean V., Jolivet P., Tournier P.-H. "A comparison of coarse spaces for Helmholtz problems in the high frequency regime." Read. https://arxiv.org/abs/2012.02678
- [Bootland21] Bootland N., Dolean V., Graham I.G., Ma C., Scheichl R. "Overlapping Schwarz methods with GenEO coarse spaces for indefinite and non-self-adjoint problems." *IMA J. Numer. Anal.* Abstract. https://arxiv.org/abs/2110.13537
- [WaveHoltz20] Appelö D., Garcia F., Runborg O. "WaveHoltz: iterative solution of the Helmholtz equation via the wave equation." *SIAM J. Sci. Comput.* Read. https://arxiv.org/abs/1910.10148
- [vanLeeuwen12] van Leeuwen T., Gordon D., Gordon R., Herrmann F.J. "Preconditioning the Helmholtz equation via row-projections." EAGE 2012. Read. https://slim.gatech.edu/Publications/Public/Conferences/EAGE/2012/vanleeuwen2012EAGEcarpcg/vanleeuwen2012EAGEcarpcg.pdf
- [GordonGordon13] Gordon D., Gordon R. "Robust and highly scalable parallel solution of the Helmholtz equation with large wave numbers." *J. Comput. Appl. Math.* 237(1). Abstract. https://cris.haifa.ac.il/en/publications/robust-and-highly-scalable-parallel-solution-of-the-helmholtz-equ/
- [DwarkaVuik20] Dwarka V., Vuik C. "Scalable convergence using two-level deflation preconditioning for the Helmholtz equation." *SIAM J. Sci. Comput.* 42(2):A901–A928. Read. https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Dwa20V.pdf
- [Sheikh13] Sheikh A.H., Lahaye D., Vuik C. "On the convergence of shifted Laplace preconditioner combined with multilevel deflation." *Numer. Linear Algebra Appl.* 20:645–662. Abstract. https://diamhomes.ewi.tudelft.nl/~kvuik/papers/She13LV.pdf
- [CBS16] Osnabrugge G., Leedumrongwatthanakun S., Vellekoop I.M. "A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media." *J. Comput. Phys.* 322:113–124. Read. https://arxiv.org/abs/1601.05997
- [Vettenburg23] Vettenburg T., Vellekoop I.M. "A universal matrix-free split preconditioner for the fixed-point iterative solution of non-symmetric linear systems." arXiv:2207.14222. Read. https://arxiv.org/abs/2207.14222
- [Stanziola25] Stanziola A., Arridge S.R., Treeby B.E., Cox B.T. "Iterative Born solver for the acoustic Helmholtz equation with heterogeneous sound speed and density." arXiv:2507.16087. Read. https://arxiv.org/abs/2507.16087
- [Stanziola21] Stanziola A., Arridge S.R., Cox B.T., Treeby B.E. "A Helmholtz equation solver using unsupervised learning: application to transcranial ultrasound." *J. Comput. Phys.* 441:110430. Read. https://arxiv.org/abs/2010.15761
- [LBS22] Stanziola A., Arridge S., Cox B.T., Treeby B.E. "A learned Born series for highly-scattering media." arXiv:2212.04948. Abstract. https://arxiv.org/abs/2212.04948
- [Azulay22] Azulay Y., Treister E. "Multigrid-augmented deep learning preconditioners for the Helmholtz equation." *SIAM J. Sci. Comput.* Read. https://arxiv.org/abs/2203.11025
- [Lerer24] Lerer B., Ben-Yair I., Treister E. "Multigrid-augmented deep learning preconditioners for the Helmholtz equation using compact implicit layers." *SIAM J. Sci. Comput.* Read. https://arxiv.org/abs/2306.17486
- [ErnstGander12] Ernst O.G., Gander M.J. "Why it is difficult to solve Helmholtz problems with classical iterative methods." LNCSE 83. Abstract and excerpt. https://archive-ouverte.unige.ch/unige:171313
- [NRT92] Nachtigal N.M., Reddy S.C., Trefethen L.N. "How fast are nonsymmetric matrix iterations?" *SIAM J. Matrix Anal. Appl.* 13(3):778–795. Abstract. https://epubs.siam.org/doi/10.1137/0613049
- [GPS19] Graham I.G., Pembery O.R., Spence E.A. "The Helmholtz equation in heterogeneous media: a priori bounds, well-posedness, and resonances." *J. Differ. Equations*. Introduction read. https://arxiv.org/abs/1801.08095
- [Shin01] Shin C., Jang S., Min D.-J. "Improved amplitude preservation for prestack depth migration by inverse scattering theory." *Geophys. Prospect.* 49(5):592–606. Abstract. https://doi.org/10.1046/j.1365-2478.2001.00279.x
- [PlessixMulder04] Plessix R.-E., Mulder W.A. "Frequency-domain finite-difference amplitude-preserving migration." *Geophys. J. Int.* 157(3):975–987. Abstract. https://repository.geologyscience.ru/items/5d0cc20e-32d0-46ae-88df-34ed96d654b6
- [Pan15] Pan W., Innanen K., Liao W. "Preconditioning for the Hessian-free Gauss-Newton full-waveform inversion." CREWES report 2015. Read. https://crewes.org/Documents/ResearchReports/2015/CRR201559.pdf
- [Demanet12] Demanet L. et al. "Matrix probing: a randomized preconditioner for the wave-equation Hessian." *Appl. Comput. Harmon. Anal.* Read. https://arxiv.org/abs/1101.3615
- [Guitton12] Guitton A., Ayeni G., Díaz E. "Constrained full-waveform inversion by model reparameterization." *Geophysics* 77(2):R117–R127. Abstract. https://doi.org/10.1190/geo2011-0196.1
- [LiuNocedal89] Liu D.C., Nocedal J. "On the limited memory BFGS method for large scale optimization." *Math. Program.* 45. Read §4. https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf
- [LiuBFGS20] Liu Q., Beller S., Lei W., Peter D., Tromp J. "Pre-conditioned BFGS-based uncertainty quantification in elastic full waveform inversion." *Geophys. J. Int.* Read. https://arxiv.org/abs/2009.12663
- [MoralesNocedal00] Morales J.L., Nocedal J. "Automatic preconditioning by limited memory quasi-Newton updating." *SIAM J. Optim.* 10(4):1079–1096. Abstract. https://doi.org/10.1137/S1052623497327854
- [Notay00] Notay Y. "Flexible conjugate gradients." *SIAM J. Sci. Comput.* 22(4):1444–1460. Abstract. https://coloradoconference.github.io/2000/abs/notay.html
- [MetivierHDR17] Métivier L. Habilitation thesis, 2017. Read one section. https://membres-ljk.imag.fr/Ludovic.Metivier/HDR/2017_METIVIER_HDR.pdf
- [MP-LBFGS26] Salvadó-Benasco M., Kssim A., Heinlein A., Krause R. "Multi-preconditioned LBFGS for training finite-basis PINNs." arXiv:2601.08709. Read. https://arxiv.org/abs/2601.08709
- [Rathore24] Rathore P., Lei W., Frangella Z., Lu L., Udell M. "Challenges in training PINNs: a loss landscape perspective." ICML 2024. Abstract. https://arxiv.org/abs/2402.01868
- [torchLBFGS] PyTorch `torch/optim/lbfgs.py` (`H_diag`, two-loop, no `H0` argument). Read. https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/lbfgs.py
- [torchspsolve] PyTorch 2.15 `torch.sparse.spsolve` docs. Read. https://docs.pytorch.org/docs/2.15/generated/torch.sparse.spsolve.html
- [cuDSS] NVIDIA cuDSS docs. Read. https://docs.nvidia.com/cuda/cudss/latest/index.html
- [AmgX] NVIDIA AmgX README. Read. https://raw.githubusercontent.com/NVIDIA/AMGX/main/README.md
- [Ginkgo] Ginkgo docs, LinOp group. Read. https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/main/group__LinOp.html
- [wavesim] wavesim docs. Read. https://wavesim.readthedocs.io/en/latest/readme.html
