# The PDE/data penalty weight in joint-space inversion

Question (#132): how do joint-space methods (penalty FWI, WRI, IR-WRI, ODIL) choose the weight between the PDE term and the data term, and what does that imply for this repo's `loss.weights` under ADR-0002's scaling? Two outputs: sweep values and a range for #80, and an answer for #86 (two weights, a ratio, or per band).

Scope: `JointFreqODIL` (`odil_wave/optimisation/joint_odil.py`), `LossCfg` and `build_optimiser` (`odil_wave/experiment/config.py`, `odil_wave/experiment/runner.py`), ADR-0001, ADR-0002, ADR-0003. It builds on [joint-block-preconditioning.md](joint-block-preconditioning.md) (**JBP**), §2 and result 2, and doesn't repeat it.

Domain terms follow `GLOSSARY.md`: **PDE term**, **Data term**, **Band**, **Warm start**, **Inverse crime**, **Independent observations**, **Wavefield block**, **Model block**, **Transducer**.

Provenance tags:

- **[S]**: established by a source (key in Sources). "(abstract)" means only the abstract or a search excerpt was reached.
- **[R]**: read from this repo's code, an issue or a research file. JBP's own pilot numbers are tagged [R: JBP].
- **[I]**: my own inference or derivation.

This session ran no solver. One arithmetic check (§4) used JBP's published numbers.

## 1. Conclusion

1. **The literature measures the PDE weight relative to the data term, and λ̃ ≪ 1 means the PDE is weak, not the data.** van Leeuwen & Herrmann write `½‖Pu−d‖² + ½λ‖A(m)u−q‖²` and scale `λ = λ̃‖PA⁻¹‖₂²`. Then "λ̃ > 1 is considered large and λ̃ < 1 is considered small" [S: vLH16 §5.3, Lemma 4.1 remark]. In their scale, this repo's default `λ̃ ≈ 2e-4` [R: JBP] is a **very weak PDE term, or equivalently a very strong data term**. It is the extended-space regime where `u` fits the data almost exactly and the wave equation takes up the misfit. **To move λ̃ toward 1 you have to lower `loss.weights.data` by about 10³–10⁴, not raise it** [I, §3]. #80's draft ("expect the best weight well above today's 1"), #86's input ("the default data weight is orders of magnitude too low") and #97's upward sweep {0.1, 1, 10, 100} all read λ̃ the wrong way round.
2. **A larger data weight can't lift the model block's reduced curvature any further.** JBP's data outliers `1 + τ·eig(Y)` run from 591 to 4746 [R: JBP result 2], so every data mode has `τy ≥ 590`. The Schur complement `S` weights each data mode by `τy/(1+τy)`, which is already 0.998 to 0.9998 of its ceiling. ×10 or ×100 on the data weight gains under 0.2% in `S`, while the wavefield-block outliers grow tenfold or a hundredfold. Lowering it by 10³ brings the outliers down to 1.6–5.8× the bulk, at the cost of 17–63% of `S` (§4) [I]. So by its curvature the default sits at the far data-dominant end. The cost there is in the conditioning of the wavefield block, not in the model's curvature.
3. **The published operating points are λ̃ ∈ [1e-4, 1] for a fixed weight, plus continuation upwards.** vLH16 compares fixed `λ̃ ∈ {0.1, 1, 10}` in all three of its examples, including a 2D ultrasound tomography case. Its continuation steps through `λ̃ = 0.1 → 100` or `1000`. Their advice is "a small fraction of ‖PA⁻¹‖₂² at the initial iterate" [S: vLH16 §6, §7]. IR-WRI uses `λ̃ = 1e-4` (inclusion), `1e-2` (Marmousi), and `1e-3` / `2e-2` (BP, noiseless / noisy) [S: Aghamiry19 §3]. It relies on Lagrange multipliers, though, to enforce the PDE, and **this repo has none**. A pure penalty keeps an `O(λ⁻¹)` error in the Lagrangian's stationarity [S: vLH16 Thm 4.2–4.3]. ODIL gives no rule: it uses unit weights [S: Karnakov22, Karnakov23]. I found no transcranial joint-space work. vLH16's 2D ultrasound example is the only USCT data point.
4. **Under the inverse crime the weight changes only the optimisation path, not the answer.** At the true model both terms are zero, whatever the weight [I]. So #80 measures conditioning and nonlinearity at a fixed L-BFGS budget. The `O(λ⁻¹)` bias of [S: vLH16 Thm 4.3] appears only with noise or **Independent observations**. There, IR-WRI raises λ̃ twentyfold for noisy data [S: Aghamiry19 §3.3].
5. **Small λ̃ is no cure for cycle skipping.** vLH16 found λ̃ = 0.1 and 1 robust to a poor initial model and λ̃ = 10 stalled [S: vLH16 §6.2.2]. Symes proves for a 1D transmission problem that WRI stays multimodal "no matter how small α may be" [S: Symes20 §0.5]. So the extended-space argument doesn't support pushing λ̃ toward 0 [I].

## 2. Literature: how the weight is chosen

| Source | Objective, weight on | Rule | Values (λ̃ = λ/‖PA⁻¹‖²) | Fixed / per frequency / per iteration |
|---|---|---|---|---|
| vLH16 | `½‖Pu−d‖² + ½λ‖Au−q‖²`, λ on PDE | λ relative to "the largest eigenvalue of A⁻ᵀPᵀPA⁻¹ at the initial iterate" (§6); a criterion "λ⁻¹‖L_v‖ small compared to ‖L_m‖", monitored, with λ increased as needed (§5.3) | fixed 0.1, 1, 10; continuation {0.1, 1, 10, 100(, 1000)}, "a few iterations" each, warm-started | fixed or ad hoc continuation; one frequency per example |
| Aghamiry19 (IR-WRI, ADMM) | `‖Pu−d‖² + λ‖Au−b‖²` plus scaled multipliers, λ on PDE | "a small fraction (1e-4) of the largest eigenvalue" of `A⁻ᵀPᵀPA⁻¹` = μ₁, following vLH16 | 1e-4 (inclusion; swept 1e-9 to 1e3); 1e-2 (Marmousi); 1e-3 / 2e-2 (BP noiseless / noisy) | "a fixed penalty parameter" throughout; multipliers make it "weakly sensitive" over moderate values; per-band recomputation of μ₁ not stated |
| vLH13 | WRI, same form | not re-read: the SLIM server refused connections this session | — | — |
| Symes20 | WRI, `α` | none; α "has the same dimensions as S"; cites Fu & Symes 2017 for a discrepancy-principle adjustment | — | dynamic (cited only) |
| Aghazade25 | extended-source AL | Residual Whiteness Principle plus robust GCV, "dynamic adjustment … during iterations" [S (abstract)] | — | per iteration |
| daSilvaYao18 | WRI, multiplicative cost | automatic trade-off by a multiplicative objective [S (abstract, via search)] | — | — |
| Karnakov22 / 23 (ODIL) | sum of squared discrete residuals plus known-value terms | no weights stated; data term is a mean over points (Karnakov23 eq. 10); only regularisation coefficients set, decaying with iterations for Newton | unit | fixed |

Three patterns [I, from the table]:

- **Normalise at the initial iterate, then fix.** vLH16 and IR-WRI both state λ as a fraction of `‖PA⁻¹‖²` at the starting model. That makes λ̃ dimensionless, and it absorbs grid spacing, frequency and source amplitude.
- **Vary it only to tighten the PDE.** vLH16's continuation raises λ̃, and IR-WRI's multipliers replace raising it. Neither lowers it. Aghamiry19's introduction says the penalty "ideally" increases over the iterations, but that its Hessian becomes ill conditioned for large λ.
- **Per frequency.** No source here varies λ̃ across frequency bands by design. IR-WRI doesn't say whether μ₁ is recomputed per batch.

## 3. The mapping between λ̃ and `loss.weights`, both ways

The objective, at one (shot `s`, frequency `f`) [R: `JointFreqODIL._terms_of_u`, module docstring]:

```
Φ = w_p · Σ_{interior+PML} |r_pde / p|² / (2 N_int n_s n_f)
  + w_d · Σ_{receivers}    |r_data / d_s|² / (2 n_r n_s n_f)
r_pde = A u − q,   p = |λ_tt| · u_ref,   r_data = P u − d,   d_s = RMS|d_obs|
```

Here `w_p = loss.weights.pde`, `w_d = loss.weights.data`, `n_r = n_transducers − 1`, and `u_ref = RMS_interior|u_warm|`. The factor 2 comes from counting real scalars (`mean_real_sq`). Dividing by `w_d / (2 n_r n_s n_f d_s²)` gives vLH16's form `‖Pu−d‖² + λ‖Au−q‖²`, per (s, f), with

```
λ[s,f]  = (w_p / w_d) · (n_r / N_int) · (d_s / (|λ_tt| u_ref))²
λ̃[s,f]  = λ[s,f] / ‖P_s A⁻¹‖₂²
        = (w_p / w_d) · (n_r / N_int) · (d_s / (|λ_tt| u_ref ‖P_s A⁻¹‖₂))²
```

This is JBP's `λ̃ = 1/(τ‖Y‖)`, with `τ = b|λ_tt|²/a` and `Y = P A⁻¹A⁻ᴴ Pᵀ` [R: JBP §2]. Both are evaluated at the warm start, `A = A(c₀)`. The inverse, to hit a target λ̃:

```
w_d / w_p = (n_r / N_int) · (d_s / (|λ_tt| u_ref ‖P_s A⁻¹‖₂))² / λ̃
          ≈ λ̃₀ / λ̃,     λ̃₀ = λ̃ at w_p = w_d = 1  (≈ 2.1e-4 in JBP)
```

| `w_d / w_p` | 10 | 1 (today) | 1e-1 | 1e-2 | 1e-3 | 2.1e-4 | 1e-4 | 1e-5 |
|---|---|---|---|---|---|---|---|---|
| λ̃ | 2.1e-5 | 2.1e-4 | 2.1e-3 | 0.021 | 0.21 | 1 | 2.1 | 21 |

Properties [I unless tagged]:

- **Only the ratio matters**, apart from the regulariser (`w_reg`, 0 by default) and L-BFGS's absolute tolerances (1e-12 / 1e-14 [R: `JointFreqODIL.__init__`], far from binding). `pde_diagonal` is invariant: `w_p` cancels from `s = √(T/H_zz)` [R: module docstring].
- **Number of sources.** `n_s` and `n_f` cancel: both terms are means over shots and frequencies, and vLH16's λ̃ is per shot too.
- **Grid spacing.** ADR-0002 was designed to make the balance independent of resolution, and JBP measured `λ̃` of 2e-4 at 2 mm and 2.1e-4 at 1 mm [R: JBP result 2]. So a fixed `w_d/w_p` already gives a fixed λ̃ across grids.
- **Number of receivers.** λ̃ goes like `n_r/‖P_sA⁻¹‖²`, and `‖P_sA⁻¹‖²` grows with `n_r`, between not at all (incoherent rows) and like `n_r` (coherent rows). Not measured.
- **Frequency.** `|λ_tt|·‖P_sA⁻¹‖` is a dimensionless Green's-function norm, and `d_s/u_ref` is a receiver-to-interior amplitude ratio. Both change with `kR` and with damping, so λ̃ at a fixed `w_d/w_p` will drift between bands by an unknown factor. **Not measured.** JBP measured λ̃ only at 80 kHz.
- **The default has changed since JBP measured λ̃₀.** Defaults are now 76 transducers and order 4 (`e357ef3`) [R: `AcquisitionCfg`]. JBP's 63 outliers per shot imply 64 transducers. λ̃₀ needs re-measuring on the current default before #80 fixes its values (§6).

## 4. Why lowering the data weight is the direction to test

JBP's closed form [R: JBP §2], with `y_i` the eigenvalues of `Y` per shot:

```
S = a Σ Jᴴ A⁻ᴴ Pᵀ (τ⁻¹I + Y)⁻¹ P A⁻¹ J,   each data mode weighted (1/y_i)·τy_i/(1+τy_i)
wavefield-block data outliers (with A(c₀)⁻¹):  1 + τ y_i
```

`τ` scales with `w_d/w_p`. Scaling JBP's measured range `τy ∈ [590, 4745]` (63 modes per shot) by a multiplier `m` on `w_d/w_p` gives the following (arithmetic check, 2026-11-27):

| `m = w_d/w_p` | λ̃ | `S` as a fraction of its ceiling, weakest to strongest mode | wavefield outliers ÷ bulk |
|---|---|---|---|
| 10 | 2.1e-5 | 0.9998 – 1.0000 | 5.9e3 – 4.7e4 |
| 1 (today) | 2.1e-4 | 0.998 – 0.9998 | 591 – 4746 |
| 1e-1 | 2.1e-3 | 0.983 – 0.998 | 60 – 476 |
| 1e-2 | 0.021 | 0.86 – 0.98 | 6.9 – 49 |
| 1e-3 | 0.21 | 0.37 – 0.83 | 1.6 – 5.8 |
| 2.1e-4 | 1 | 0.11 – 0.50 | 1.1 – 2.0 |
| 1e-5 | 21 | 0.006 – 0.045 | ≈ 1 |

Reading [I]:

- **Upwards** (`m > 1`): `S` gains nothing, and the wavefield block gets up to 10× more ill conditioned per decade. #97's prediction ("c recovery improves monotonically with the data weight") should fail above 1, and the curve should be flat or get worse.
- **Down to 1e-2**: `S` keeps 86% or more while the outliers fall 100×. At a fixed L-BFGS budget this is the most likely place for a gain.
- **1e-3 to 2e-4 (λ̃ 0.2–1, vLH16's range)**: the outliers have almost gone, `S` drops 2–9× on the weakest modes, and the problem moves toward reduced FWI, which is more nonlinear from a homogeneous start [S: vLH16 §6.2.1, fig. 6].
- **Below 1e-4 (λ̃ > 2)**: `S → aτGᴴG → 0` relative to `H_zz`. The case-3 stall should get worse.

Caveats [I]: this is the Gauss-Newton Hessian at the warm start, per shot, on the pre-`e357ef3` default. L-BFGS with history 20 isn't Newton. It says nothing about nonlinearity along the path, nor about #111's transducer collapse.

## 5. Answers

### #80: sweep values and range

- **Variable:** `loss.weights.data`, with `loss.weights.pde = 1` held fixed. That keeps the PDE term's magnitude, ADR-0002's balance check and the regulariser's balance where they are.
- **Values** (log-spaced and centred below today's value, about half a decade apart at the interior):
  `w_d ∈ {10, 1, 1e-1, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}`, which is λ̃ ≈ {2e-5, 2e-4, 2e-3, 0.02, 0.07, 0.2, 0.7, 2, 21}.
  - 10 is the one upward control point, to confirm the flat or worse prediction of §4.
  - 1e-5 is the λ̃ ≫ 1 end, where the FWI-like stall should return.
  - If the budget allows only 6 runs: {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5}.
- **Expected best:** `w_d` between 1e-3 and 1e-1 (λ̃ about 2e-3 to 0.2). The outliers have mostly gone by 1e-2 while `S` is still above 85%. If the sweep finds an interior optimum, add points at `3e-2` and `3e-3`. If the best is at an end, extend by a decade.
- **Link 2 (250²):** expect the same best `w_d`, because λ̃ is grid independent [R: JBP]. The L-BFGS budget and conditioning can still move it (#80's own caveat).
- **Report** λ̃ per (shot, frequency) at the warm start alongside each run, so results transfer to other frequencies and configs in λ̃ terms.
- **#97's scratch sweep** should extend downward to 1e-4 too. It is cheap, and it tests §4 directly before #80 is finalised.

### #86: how to represent the balance

Options:

1. **Two weights `{pde, data}` as today.** One owner per weight (ADR-0001). Only the ratio matters, but `w_p` also sets the regulariser balance and the PDE term's absolute scale on the tape.
2. **One ratio** (#123). It is strictly less expressive than option 1 with `w_p = 1`, and it would change every config. It adds nothing, because sweeping `w_d` at `w_p = 1` is already a ratio sweep.
3. **A target λ̃.** The data weight would be derived once per band at the warm start, from `w_d = w_p·λ̃₀(s, f)/λ̃`. This is the literature's own convention (normalise at the initial iterate, then fix) [S: vLH16; Aghamiry19]. It fits ADR-0002's pattern of fixed per-band scales computed at the warm start, and it keeps the weight constant within a band, so the L-BFGS history stays valid. It needs `‖P_sA(c₀)⁻¹‖₂` per shot: a few power iterations reusing the warm-start factors, which ADR-0003 already keeps for the band. It would amend ADR-0002.
4. **Per-band weight** (#86's (a) multiplier, (b) override or (c) none). A per-band multiplier is the only form of vLH16's continuation this repo can run: repeat a frequency in successive bands with rising `pde_weight_factor`. That is because the weights must stay fixed within a band ("never adaptive" [R: #86]).

**Recommendation** [I]:

- **Keep two weights (option 1), and close #123's ratio as not needed.**
- **Take no per-band weight now (option c).** #80 sweeps `w_d` alone.
- **Open a ticket to log λ̃ per (band, frequency).** Decide on option 3 when it lands. If λ̃ at a fixed `w_d/w_p` drifts by more than about 3× across the bands of #143's schedule, adopt option 3 with an ADR-0002 amendment, not per-band overrides. A λ̃ target is what transfers across frequency and acquisition.
- **Keep option 4(a)** only as a vehicle for a later continuation experiment, once #80 has found the fixed optimum.

## 6. Open questions and suggested tickets

1. **Re-measure λ̃₀ on the current default** (76 transducers, order 4), per frequency at 40, 80 and 160 kHz, and at two resolutions. This is JBP §8's report item 1, so it fits inside #131. It is a prerequisite for fixing #80's values in λ̃ terms.
2. **λ̃ as a tape diagnostic** at each band's warm start (new feature ticket, feeds #86 option 3 and #143).
3. **#97's prediction should be revised**: from "monotone improvement with the data weight" to "flat or worse above 1; any gain comes below 1". Post this on #97.
4. **Multipliers in the joint solve.** IR-WRI's fixed small λ̃ works because of its multiplier updates [S: Aghamiry19]. Updating a multiplier on `r_pde` between bands (not within one, to keep L-BFGS valid) doesn't eliminate `u`, so ADR-0003 may not rule it out. Possible grilling ticket after #80.
5. **λ̃ under independent observations.** There the weight also sets the `O(λ⁻¹)` model bias [S: vLH16 Thm 4.3], and IR-WRI raises λ̃ for noise [S: Aghamiry19]. #80's inverse-crime optimum may not transfer. Flag it on the independent-observations work.

## 7. Unverified and limits

- vLH13 wasn't re-read: the SLIM server refused connections. Its values aren't cited here.
- Aghazade25 and daSilvaYao18 were reached as abstracts or search excerpts only.
- Aghamiry19's Marmousi section was read only in part (truncated). Whether μ₁ is recomputed per frequency batch isn't stated in what I read.
- All of §4 rests on JBP's pilot numbers on the older default. No solver was run in this session.

## Sources

- [vLH16] van Leeuwen T., Herrmann F.J. "A penalty method for PDE-constrained optimization in inverse problems." *Inverse Problems* 32:015007 (2016). arXiv:1504.02249. Read: §3–7, Lemma 4.1 remark, Theorems 4.2–4.3, §5.3 "Choosing λ", §6 case studies and tables 5–7. https://arxiv.org/abs/1504.02249
- [Aghamiry19] Aghamiry H.S., Gholami A., Operto S. "Improving full-waveform inversion by wavefield reconstruction with the alternating direction method of multipliers." *Geophysics* 84(1):R139–R162 (2019). arXiv:1809.00891. Read: §1–3. https://arxiv.org/abs/1809.00891
- [Symes20] Symes W.W. "Wavefield reconstruction inversion: an example." *Inverse Problems* 36(10) (2020). arXiv:2003.14181. Read: §0.4–0.7. https://arxiv.org/abs/2003.14181
- [Karnakov22] Karnakov P., Litvinov S., Koumoutsakos P. "Optimizing a DIscrete Loss (ODIL) to solve forward and inverse problems for partial differential equations using machine learning tools." arXiv:2205.04611; *PNAS Nexus* 3(1) (2024). Read (examples' loss definitions). https://arxiv.org/abs/2205.04611
- [Karnakov23] Karnakov P., Litvinov S., Koumoutsakos P. "Flow reconstruction by multiresolution optimization of a discrete loss with automatic differentiation." arXiv:2303.04679. Read (§3 loss definitions). https://arxiv.org/abs/2303.04679
- [Aghazade25] Aghazade K., Zand T., Gholami A. "Automatic penalty parameter selection by Residual Whiteness Principle (RWP) and GCV for full waveform inversion." arXiv:2512.16757 (2025). Abstract. https://arxiv.org/abs/2512.16757
- [daSilvaYao18] da Silva N.V., Yao G. "Wavefield reconstruction inversion with a multiplicative cost function." *Inverse Problems* 34:015004 (2018). Search excerpt only. https://ui.adsabs.harvard.edu/abs/2018InvPr..34a5004D/abstract
- [vLH13] van Leeuwen T., Herrmann F.J. "Mitigating local minima in full-waveform inversion by expanding the search space." *Geophys. J. Int.* 195:661–667 (2013). Not reached this session.
