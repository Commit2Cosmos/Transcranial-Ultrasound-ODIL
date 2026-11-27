# Does a larger data weight unstick case 3? A scratch sweep for #97

Question: is #93's case 3 (true wavefield, homogeneous initial model, `model_preconditioner: none`, regulariser off, inverse crime) the #33 stall, caused by a data term too weak to pin `u`? Prediction under test, from #97's 2026-11-14 scope: **"c recovery improves monotonically with the data weight, and the wavefield drifts less."** Scope: #97, #93, #33, [joint-block-preconditioning.md](joint-block-preconditioning.md) (**JBP**) results 1–2, #80, #86.

Domain terms follow `GLOSSARY.md`: **Model block**, **Wavefield block**, **Initial model**, **Warm start**, **True model**, **Inverse crime**, **Interior**, **PDE term**, **Data term**, **Scratch run**.

Provenance tags:

- **[P]**: measured in a scratch run in this session (scripts under `research/scratch/`, uncommitted when run). It indicates, it does not establish. Under the glossary's rule for scratch runs, no experiment record may cite it.
- **[R]**: read from this repo's code, issues or research notes.
- **[S]**: established by a source, here only van Leeuwen & Herrmann 2016 as JBP reports it.
- **[I]**: my own inference.

## 1. Verdict

**The prediction is refuted.** The data-weight explanation of the stall is wrong in its actionable form: raising `w_d` doesn't help.

1. **c recovery is flat in `w_d` over two to three decades, then gets worse.** Unit seam, 200 iterations: c error 1.250–1.256% for every `w_d` in {0.1, 1, 10, 100}, rising to 1.46% at 1e3 and 1.86% at 1e4 [P]. E2E seam, 50 iterations: 3.35–3.36% at `w_d` ∈ {1, 10}, 3.46% at 100, 3.98% at 1e3, 4.25% at 1e4 [P]. Recovery improves with `w_d` only **below** about 0.1 (unit) or 1 (e2e).
2. **The wavefield's drift isn't monotone in `w_d` either.** At the plateau the unit wavefield error ends at 0.39–0.40% after 200 iterations. It is 3.0% at `w_d` = 1e-3 and 4.0% at 1e4 [P]. Raising `w_d` pins the receivers harder (the relative data residual falls about 10× per decade), but the interior drifts more above `w_d` ≈ 100 [P].
3. **The Schur complement `S` saturates in `w_d`.** This is the mechanism. `median diag(S)/H_zz` rises from 0 at `w_d` = 0 (JBP result 1, reproduced: 6e-30) to 0.72% at 1, and is flat from 10 upwards: 0.80%, 0.81%, 0.81%, 0.81% for 10, 100, 1e3, 1e4 (unit, at the truth) [P]. The e2e seam gives the same shape, plateauing at 0.79% [P]. This is JBP §2's large-`τ‖Y‖` limit, `S → a GᴴY⁻¹G`, which doesn't depend on `τ ∝ w_d/w_p` [S, via JBP; I]. Past the knee, more data weight can't lift `S`.
4. **What limits c is the rank of `S`, not its size.** Even saturated, only 27–31 of 400 eigenvalues `μ` of `H_zz⁻¹ᐟ²SH_zz⁻¹ᐟ²` exceed 1e-2, with `μ_max` 0.36–0.47 and a median `μ` at round-off [P]. With `u` free, the 8-transducer data constrain only about 30 model directions per frequency, as JBP's `rank(S) ≤` (real data count) bound predicts [R: JBP §2; I]. The frozen-`u` recovery in #93 (c to 2.6e-6) works because fixing `u` everywhere supplies information the joint problem doesn't have [I]. So "the model is identifiable" holds only given the whole wavefield, not given receiver data.
5. **#97's reading of `λ̃` is inverted.** In van Leeuwen & Herrmann's penalty, `λ` weights the **PDE** term, so `λ̃ ≈ 2e-4 ≪ 1` means the data term already dominates the wavefield block at the receivers. It doesn't mean the data term is too weak [S, via JBP §2's `τ‖Y‖ ≈ 4700`; I]. JBP's own wording ("the second limit") is consistent with this; the ticket's paraphrase isn't.

Part of the original case-3 failure **no longer reproduces at HEAD** (§4). At data weight 1 the unit seam now reaches 1.25% (#93: 2.62%), with the wavefield returning to 0.39% (#93: 3.8% and growing).

## 2. Results

Errors are relative L2 over the interior, the wavefield over all shots and frequencies, against the true model and the true wavefield. "Φ" is the weighted objective at the iterate. It scales with `w_d`, so compare it only within a row group. The PDE and data terms are unweighted. Iteration 0 is the same for every weight: unit c 3.253e-2, e2e c 4.371e-2, `u` error 7e-17, data term 0.

### Unit seam (`truth_problem` of `tests/test_joint_odil.py`, 50 / 200 outer × 20 inner)

| `w_d` | c err @50 | c err @200 | u err @50 | u err @200 | Φ @50 | Φ @200 | PDE term @200 | rel data res @200 |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 2.56e-2 | 1.73e-2 | 7.38e-2 | 2.97e-2 | 3.7e-6 | 4.9e-7 | 6.8e-9 | 3.1e-2 |
| 0.01 | 1.75e-2 | 1.41e-2 | 2.84e-2 | 1.33e-2 | 8.4e-6 | 3.2e-7 | 3.8e-8 | 7.6e-3 |
| 0.1 | 1.53e-2 | **1.250e-2** | 1.43e-2 | 4.11e-3 | 6.1e-6 | 8.3e-8 | 4.2e-8 | 9.1e-4 |
| 1 | 1.46e-2 | **1.253e-2** | 1.10e-2 | 3.95e-3 | 4.8e-6 | 1.0e-7 | 8.6e-8 | 1.7e-4 |
| 10 | 1.46e-2 | **1.254e-2** | 1.10e-2 | 4.00e-3 | 4.9e-6 | 1.3e-7 | 1.3e-7 | 2.3e-5 |
| 100 | 1.50e-2 | **1.256e-2** | 1.39e-2 | 4.04e-3 | 9.2e-6 | 1.5e-7 | 1.5e-7 | 2.4e-6 |
| 1000 | 2.24e-2 | 1.46e-2 | 4.96e-2 | 1.11e-2 | 2.6e-4 | 5.1e-6 | 5.1e-6 | 1.9e-6 |
| 10000 | 3.04e-2 | 1.86e-2 | 3.04e-2 | 4.02e-2 | 7.5e-4 | 1.4e-4 | 1.4e-4 | 4.9e-6 |

Wall time: about 2.3 s for 50 iterations and 9–11 s for 200 per weight, with 1069–1113 / 4418–4789 closures [P].

### E2E seam (`configs/checks/tiny.yaml`, 10 / 50 outer × 20 inner)

| `w_d` | c err @10 | c err @50 | u err @10 | u err @50 | Φ @10 | Φ @50 | PDE term @50 | rel data res @50 |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 4.09e-2 | 3.96e-2 | 5.28e-2 | 6.74e-2 | 7.0e-5 | 2.7e-6 | 5.4e-8 | 7.3e-2 |
| 0.01 | 4.07e-2 | 3.48e-2 | 5.24e-2 | 3.35e-2 | 8.4e-5 | 4.6e-6 | 1.6e-6 | 2.4e-2 |
| 0.1 | 4.04e-2 | 3.40e-2 | 4.59e-2 | 2.34e-2 | 1.6e-4 | 7.4e-6 | 5.1e-6 | 6.7e-3 |
| 1 | 3.98e-2 | **3.358e-2** | 4.60e-2 | 2.03e-2 | 2.2e-4 | 8.0e-6 | 7.5e-6 | 9.1e-4 |
| 10 (tiny's default) | 3.97e-2 | **3.352e-2** | 4.47e-2 | 2.02e-2 | 2.3e-4 | 8.1e-6 | 8.1e-6 | 1.0e-4 |
| 100 | 4.15e-2 | 3.46e-2 | 3.98e-2 | 2.89e-2 | 3.5e-4 | 2.0e-5 | 2.0e-5 | 1.7e-5 |
| 1000 | 4.33e-2 | 3.98e-2 | 1.71e-2 | 4.55e-2 | 7.3e-4 | 2.3e-4 | 2.3e-4 | 2.7e-5 |
| 10000 | 4.36e-2 | 4.25e-2 | 8.70e-3 | 3.04e-2 | 9.8e-4 | 5.3e-4 | 5.3e-4 | 5.1e-6 |

Wall time: about 0.35 s for 10 iterations and 1.7–2.3 s for 50 [P].

At 10 iterations a large `w_d` keeps `u` closest to the true wavefield (0.87% at 1e4). It does that by freezing the whole solve: c barely moves (4.37% → 4.36%), and by 50 iterations `u` has drifted further than at the plateau [P].

### Model-block Schur complement against `w_d` (Gauss-Newton, `research/scratch/schur_vs_data_weight.py`)

| `w_d` | unit, truth: median diag(S)/H_zz | μ_max | #μ > 1e-2 | unit, case-3 start: median | μ_max | e2e, truth: median | μ_max | #μ > 1e-2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 6e-30 | 3e-29 | 0 | 6e-30 | 4e-29 | 3e-30 | 0 | 0 |
| 1e-3 | 1.10e-4 | 0.006 | 0 | 1.19e-4 | 0.007 | 1.41e-4 | 0.007 | 0 |
| 1e-2 | 9.54e-4 | 0.052 | 13 | 9.96e-4 | 0.059 | 1.21e-3 | 0.060 | 16 |
| 0.1 | 4.21e-3 | 0.212 | 21 | 4.15e-3 | 0.250 | 5.05e-3 | 0.263 | 29 |
| 1 | 7.21e-3 | 0.331 | 25 | 7.06e-3 | 0.428 | 7.49e-3 | 0.407 | 31 |
| 10 | 8.02e-3 | 0.355 | 27 | 7.85e-3 | 0.471 | 7.88e-3 | 0.431 | 31 |
| 100 | 8.09e-3 | 0.358 | 27 | 7.96e-3 | 0.476 | 7.93e-3 | 0.434 | 31 |
| 1e3 | 8.10e-3 | 0.358 | 27 | 7.97e-3 | 0.476 | 7.94e-3 | 0.434 | 31 |
| 1e4 | 8.10e-3 | 0.358 | 27 | 7.97e-3 | 0.476 | 7.94e-3 | 0.434 | 31 |

400 model cells at both seams. The e2e case-3 start is within 8% of the e2e truth column at every weight, with counts within one [P].

`S` is formed as `Σ_s (Π⊥_s J_z)ᵀ(Π⊥_s J_z)`, with `J` the dense per-shot Jacobian of the production residuals (`JointFreqODIL._residuals_of_u`, scaled as in `_terms_of_u`) and `Π⊥_s` the projector off `range(J_u,s)` [P]. This is the Gauss-Newton `S`, exact at the truth where the residuals vanish. `H_zz` is the PDE term's model diagonal.

## 3. Reading

- **The knee sits where `τ‖Y‖ ≈ 1`** [I]. Below it, `S ≈ aτGᴴG` grows linearly in `w_d` (the 1e-3 → 1e-2 step is 8.7×), and so does c recovery. Above it, `S` is fixed and c recovery is flat. On these 20² grids the knee is at `w_d` ≈ 0.1–1. JBP's `τ‖Y‖ ≈ 4700` at `w_d` = 1 puts the default 125² problem's knee near `w_d` ≈ 2e-4, so the default problem is deep in the plateau [R: JBP §2; I]. That is the regime that matters for #80.
- **Degradation above `w_d` ≈ 100 is an optimiser effect** [I]. `S` is unchanged there, but the data outliers of the wavefield block, `1 + τ eig(Y)` (JBP result 2), grow linearly in `w_d`. L-BFGS spends its budget on them: the PDE term at the final iterate is 30–1000× the plateau's [P], so the solve is less converged, not stuck in a different valley.
- **The stall at the plateau is an information limit** [I]. About 30 well-conditioned model directions out of 400 means most of the c error lies where the joint objective is nearly flat in `c` once `u` relaxes. A preconditioner or a weight can't fix that, but more data (transducers, frequencies), a regulariser, or a better initial model can. `pde_diagonal`'s case-3 success in #93 (c to 0.39%) points the other way: it suggests the plateau's error isn't all in `null(S)` and that optimiser scaling matters too. The decomposition that would settle this wasn't run (§5).

## 4. Does #93 still reproduce?

Not fully. At `w_d` = 1, the unit seam's c error after 200 iterations is 1.25% at HEAD against 2.62% at `78a0807`, and the wavefield error 0.39% against 3.8% [P, R: #93]. The e2e starting error differs (4.37% against 5.56%) because the tiny truth changed to Shepp-Logan, scored on the whole model (#168, `5890eb6`) [R]. Changes between the two commits that touch the operator or the solve include the stretched-coordinate PML (#37, `12ae721`), dividing the PML's rows by `s_x s_y` (#165, `1d66417`), the raw source in the residual (#165, `30de1d9`) and edge-velocity replication into the PML (#164, `a543d2f`) [R: `git log 78a0807..HEAD`]. I didn't bisect. The qualitative failure, c far from the frozen-`u` 2.6e-6, still holds.

## 5. What #80 and #86 should take from this

- **#80 (default data weight):** a larger `w_d` won't fix the model-block stall. Under the inverse crime, model recovery is flat across the plateau above the `τ‖Y‖ ≈ 1` knee, and degrades once the data outliers dominate the wavefield block. On these grids that's about 2.5 decades wide (0.1–100). If #80's metrics follow the same shape, its sweep should bracket both edges: the knee (default problem: near `w_d` ≈ 2e-4 at `w_p` = 1) and the degradation edge. It should expect a flat `rmse_brain` between them, and so a default chosen inside the plateau mostly for wavefield-block conditioning (lower is better). Its draft also echoes the "`λ̃` far below the literature" framing. As §1.5 says, small `λ̃` means the data already dominates. Whether the plateau carries over from 20² to 125² and 250² is #80's Link 2 question. This sweep says only that the edges should scale with `τ‖Y‖`.
- **#86 (per-band PDE weight):** `S` depends on the weights only through `τ ∝ w_d/w_p`. The overall scale enters only through L-BFGS's absolute tolerances [I]. So representing the balance as one ratio per band loses nothing for the model block. A per-band weight changes model recovery only if a band crosses the knee or the degradation edge. Using it to fight the stall isn't supported.
- **#33 / ADR-0003:** case 3 is the stall JBP result 1 describes (`S = 0` without the data term, reproduced here exactly). But its residual is a rank limit, not a weight limit.

## Open questions and candidate tickets

1. **Is the plateau's remaining c error in `null(S)`?** Project the case-3 c error at 200 iterations onto the eigenvectors of `S` with `μ > 1e-2`, with and without `pde_diagonal`. This separates the information limit from optimiser scaling, and bears on #113. Scratch, minutes.
2. **What weakened the case-3 stall between `78a0807` and HEAD?** Bisect over the commits in §4. Scratch, about 30 s per step. This matters if #100 pins case 3 as a test with #93's numbers.
3. **Amend the `λ̃` wording** in #97's and #80's bodies, and JBP result 2's "puts a number on #33's '`u` absorbs the misfit'". The data term isn't weak relative to the PDE term.
4. **Does the degradation edge sit where `τ‖Y‖ · κ(H_uu)` passes some threshold?** This would let #80 predict the edge per grid without a sweep.

## Reproduction

Repo sha `5b3b236` (branch `research/data-weight-sweep` adds only this note and the two scripts). float64, CPU, `uv run`, inverse crime, regulariser off, `model_preconditioner: none`, `wavefield_preconditioner: none`. Every invocation ran alone. #185 forbids parallel runs, and the one accidental overlap (two copies of the deterministic Schur script, unit seam) affected only wall time, which isn't reported for that script.

```
uv run python research/scratch/data_weight_sweep.py unit 0.1 1 10 100 1000 10000
uv run python research/scratch/data_weight_sweep.py unit 0.001 0.01
uv run python research/scratch/data_weight_sweep.py e2e 0.001 0.01 0.1 1 10 100 1000 10000
uv run python research/scratch/schur_vs_data_weight.py unit
uv run python research/scratch/schur_vs_data_weight.py e2e
```

- **Case 3 at both seams:** the band is built from the homogeneous initial model, then `band.u_init` is replaced by the true wavefield (`band.true_wavefields`) before `JointFreqODIL` is built. So `u_ref` and every residual scale come from the true wavefield, as in #93's monkeypatch of the warm start.
- **Unit seam:** the `truth_problem` setup (overdensity +120 m/s, r = 12 mm, 40 kHz, 8 transducers, 12-cell PML), `JointFreqODIL` with `pde_weight=1`, `data_weight=w_d`, `reg_weight=0`, `inner_max_iter=20`, `history_size=20`, strong Wolfe, `tolerance_grad=1e-12`, `tolerance_change=1e-14`. One 200-iteration run per weight, read at 50 and 200. The outer loop has no schedule, so iterate 50 of that run equals a 50-iteration run.
- **E2E seam:** `configs/checks/tiny.yaml` deep-merged with `{loss: {weights: {pde: 1, data: w_d, reg: 0}}, optimiser: {n_iter: 50, log_every: 1, model_preconditioner: none}}`, then built through the run's own `build_problem`, `Problem.make_band` and `build_optimiser`. Artifact writing is skipped (no `run_inverse`). Read at 10 and 50.
