# Primary-source thresholds for the discretisation and acquisition constraints

Question (#135): which thresholds do primary sources support for the constraints in #124's table, which are rules of thumb today, and how severe is each violation under the inverse crime and once observations are independent (#136)? Rows are numbered as in #124. Scope: rows 5 (dispersion), 6 (ring sampling), 11 (PML thickness) and 12 (skull thin layer) in depth, plus a severity line for every row. Consumers: #124 (warnings), #138 (defaults).

Domain terms follow `GLOSSARY.md`: **Transducer**, **Shot**, **Interior**, **PML**, **Inverse crime**, **Skull region**.

Provenance tags, as in [near-source-artefacts-and-fringes.md](near-source-artefacts-and-fringes.md) (**NSA**):

- **[S]**: established by a source (key in Sources). "(abstract)" means only the abstract or a search excerpt was reached.
- **[R]**: read from this repo's code or issues.
- **[I]**: my own inference or derivation.
- **[C]**: computed by the script in the appendix, from the repo's own stencil coefficients.

Default config used throughout [R: `config.py`, `grid.py`, `velocity_models.py`, measured with `build_problem(RunConfig())`]: 80 kHz single band; interior 125² nodes over 0.25 m, so `h` = 2.016 mm; order 2; `dt` = 500 µs / 1299 = 0.385 µs; 64 transducers on a circle of radius `R` = 0.9 × 0.125 = 112.5 mm; PML 40 cells (code default; #124's study used 30); water 1500 m/s, brain 1450–1750 m/s, skull rim 3000 m/s; head semi-axes 74.6 × 98.8 mm, largest radius `r0` = 99.8 mm; skull rim 2 phantom pixels ≈ 4.0 mm.

## 1. Conclusion

| # | Threshold | Basis | Default | Severity, inverse crime | Severity, independent observations |
|---|---|---|---|---|---|
| 5 | Worst-direction phase-velocity error `e` from the stencil symbol; accumulated phase `2π(2R/λ)·e ≤ ε` over the ring diameter. `ε` = π/4 reproduces #124's numbers; Kreiss's own table uses 0.1 rad | Method and `D/λ` scaling: [S: KO02, AKB74, BGT85]. **ε is a judgement call**, in the sources too | o2: 1.51 rad over 2R, `e` = 2.0% (30 m/s) ✗ | **info** | **warn** |
| 6 | `n_tx > 2k·r0` with `r0` the radius of everything that scatters or is inverted, concentric with the ring. `r0 = R` gives the λ/2-pitch rule `2kR` | [S: Sim07 eq. 32–33]. Choice of `r0` here: [I] | 64 vs 67 (`r0` = head) / 76 (`r0` = ring) ✗ | **warn** | **warn** |
| 11 | No primary threshold for this layer type; λ/2 at `f_min` is a judgement call | PML theory doesn't apply: the layer is graded damping (#37). Defer to #134 | 4.3λ (PML 40) ✓ | **info** (modelled identically in data) | **warn** (also when `pml.observation_width` is set) |
| 12 | Thickness `T` vs local λ: below λ/8 (Widess) only `T × contrast` is constrained; separating the surfaces needs ≈ λ/4 (reflection) or ≈ 0.31λ (full-ring FWI point-spread first null); thickness and speed separately ≳ λ/2 | λ/8: [S: Wid73 (secondary)]; λ/4: [S: KW82 (abstract)]; 0.31λ and λ/2: [I] from the 2k wavenumber disc [S: VO09 via NSA] | `T` = λ_skull/9.4, λ_water/4.7 ✗ | **info** (resolution report, truth-dependent) | **info** |

Four results:

1. **#124's dispersion numbers are right, and order 4 is enough at the default grid even under a strict tolerance.** Redone from the repo's `FD2_COEFFS` symbol [C]: at `D/λ` = 12 (the 225 mm ring diameter at 80 kHz) and ε = π/4, the required points per wavelength are 12.7 / 5.3 / 4.0 / 3.4 / 3.1 for orders 2 / 4 / 6 / 8 / 10. Under Kreiss's 0.1 rad tolerance they are 35.3 / 8.9 / 5.7 / 4.6 / 4.0. The default grid gives 9.3 (9.0 at the 1450 m/s brain minimum), so order 4 passes both and order 2 fails both (§2).
2. **The two #124 ring rules are one rule with different radii.** `2πR/(λ/2)` = `2kR`. Simonetti et al. derive `N > 2k·r0`, where `r0` circumscribes the object, and show it reduces to the λ/2 pitch when `r0 = R` [S: Sim07 eq. 32–33]. They also warn that aliasing can occur "when r0 approaches R ... even if (33) is satisfied". Here `r0/R` = 0.89, and the inversion updates `c` right up to the transducers (NSA §1), so the conservative `2kR` is the one to check. That needs 76 transducers at 80 kHz, not 100 (§3).
3. **Ring under-sampling is not removed by the inverse crime.** It is a property of the data, not of how the data were produced. Dispersion (row 5), time-symbol error (row 9) and PML reflections (row 11) are reproduced exactly when observations come from the inversion's own operator, so they are harmless there [I]. #124's "inverse crime makes #5/#6 harmless" holds for #5 only.
4. **At the default `dt`, the time symbol dominates the dispersion error for every order above 2.** The `sin²` time symbol shortens the numerical wavenumber by `(ωdt)²/24` = 0.16% (2.3 m/s) whatever the spatial order. Order 4's spatial error at 9.3 points per wavelength is 0.11% with the opposite sign [C]. Once observations are independent, #116 (exact time symbols) matters as much as the stencil order.

## 2. Row 5: numerical dispersion

### 2.1 Method

The residual is `λ_tt u − c'²∇'²u − f'` with `λ_tt = −(4/Δt'²) sin²(ωΔt'/2)` [R: `operator/utils.py`]. The Laplacian is a cross stencil built from 1D central coefficients `FD2_COEFFS` [R: `operator/spatial.py`]. A plane wave `exp(iκ·x)` solves the homogeneous discrete equation when

`s(κh cosφ) + s(κh sinφ) = (ω_eff h / c)²`, with `s(θ) = −(c₀ + 2Σ c_m cos mθ)` and `ω_eff = (2/Δt) sin(ωΔt/2)` [I].

The relative wavenumber error `e = κ/k − 1` equals the relative phase-slowness error. Over a path of `N = D/λ` wavelengths it accumulates a phase error `2πN·e`. This is the analysis Kreiss uses: "a phase error e = (1 − α)ωt", tabulated as points per wavelength for a 10% or 1% error "after calculating during q time periods" [S: KO02 §4, Table 1]. I checked that the 10% row matches `e` = 0.1 rad: for the second-order first-derivative stencil, `1 − α ≈ θ²/6` gives 20.3 `q^½` points, as tabulated [I]. The required points per wavelength therefore grow as `(D/λ)^(1/p)` for order `p` [S: KO02 Table 1]. Bayliss et al. state the same for Helmholtz-type problems: points per wavelength alone don't fix accuracy, and `k³h²` controls it at second order [S: BGT85 (abstract)]. Alford, Kelly and Boore's rule of thumb, "of the order of ten or more" points per upper half-power wavelength at second order and "five or more" at fourth order, is the fixed-distance version from seismograms [S: AKB74 abstract].

Two choices are mine [I]:

- **Path length `D` = ring diameter `2R`** (225 mm), not the domain width: the data only see source-to-receiver paths, and the longest is diametric. The issue's 13.4 at order 2 is the same calculation over 0.25 m (`D/λ` = 13.3).
- **Speed = slowest along the path.** Water 1500 m/s gives 9.30 points per wavelength; the brain minimum 1450 m/s gives 9.0. The skull is faster and less dispersive.

The cross stencil is most dispersive along the grid axes, so the axis direction sets the worst case [C].

### 2.2 Results

At the default grid (9.30 points per wavelength, `ωdt` = 0.194) [C]:

| order | `e` axis | `e` diagonal | worst `e` incl. time symbol | phase over 2R (spatial) | velocity bias at 1500 m/s |
|---|---|---|---|---|---|
| 2 | +2.0e-2 | +9.8e-3 | +1.8e-2 | 1.51 rad | 30 m/s |
| 4 | +1.1e-3 | +2.8e-4 | −1.3e-3 | 0.084 rad | 1.7 m/s |
| 6 | +7.9e-5 | +1.0e-5 | −1.6e-3 | 0.006 rad | 0.1 m/s |
| 8, 10 | < 1e-5 | | −1.6e-3 | < 0.001 rad | < 0.01 m/s |

The time symbol alone gives `e` = −1.56e-3 (−2.3 m/s) [C].

Points per wavelength needed, spatial only, worst direction [C]:

| ε | D/λ | o2 | o4 | o6 | o8 | o10 |
|---|---|---|---|---|---|---|
| π/4 | 5 | 8.4 | 4.2 | 3.4 | 3.0 | 2.8 |
| π/4 | 10 | 11.7 | 5.0 | 3.8 | 3.3 | 3.1 |
| π/4 | **12** (80 kHz) | **12.7** | **5.3** | **4.0** | 3.4 | 3.1 |
| π/4 | **15** (100 kHz) | **14.2** | **5.6** | **4.1** | 3.5 | 3.2 |
| π/4 | 20 | 16.4 | 6.0 | 4.3 | 3.7 | 3.3 |
| π/4 | 30 | 20.0 | 6.7 | 4.7 | 3.9 | 3.5 |
| π/4 | 50 | 25.7 | 7.6 | 5.1 | 4.2 | 3.7 |
| 0.1 rad | 5 | 22.8 | 7.1 | 4.9 | 4.0 | 3.6 |
| 0.1 rad | 10 | 32.2 | 8.5 | 5.5 | 4.5 | 3.9 |
| 0.1 rad | **12** | **35.3** | **8.9** | **5.7** | 4.6 | 4.0 |
| 0.1 rad | **15** | **39.4** | **9.4** | **5.9** | 4.7 | 4.1 |
| 0.1 rad | 20 | 45.5 | 10.1 | 6.2 | 4.9 | 4.2 |
| 0.1 rad | 30 | 55.7 | 11.2 | 6.7 | 5.2 | 4.4 |
| 0.1 rad | 50 | 71.9 | 12.8 | 7.3 | 5.6 | 4.7 |

These reproduce #124's 13.4 / 5.4 / 4.0 at 80 kHz (over 0.25 m) and ~15 at 100 kHz. At 100 kHz the default grid gives 7.4 points per wavelength: order 4 passes π/4 (5.6) but fails 0.1 rad (9.4); order 6 passes both (5.9).

### 2.3 Tolerance

No source fixes ε for this problem; Kreiss tabulates two (10% and 1%) and leaves the choice to the user [S: KO02 Table 1]. Two ways to anchor it [I]:

- **Cycle skipping.** A phase error of π (half a period) is where a local misfit picks the wrong cycle (VO09, not re-read this session). π/4 is a factor-4 margin below that. It doesn't account for the starting-model error that adds to it.
- **Velocity bias.** With independent observations, a uniform phase-slowness error `e` is absorbed by the inversion as a speed bias of about `e·c`, independent of distance. The phantom's brain contrasts are tens of m/s (`SOFT_SLOPE` = 300 m/s times Shepp-Logan steps of 0.1–0.2) [R: `velocity_models.py`]. So 30 m/s (order 2) would be comparable to the target contrasts, and 2 m/s (order 4, or the time symbol) would not.

Recommendation for #124: compute `e` directly from the symbol (as the appendix does, about 15 lines with a root solve) rather than hard-coding a points-per-wavelength table. Report both the phase over `2R` against ε = π/4 and the bias `e·c_min` in m/s. Mark ε as a judgement call in the message.

## 3. Row 6: ring angular sampling

**The rule.** Simonetti, Huang and Duric analyse a ring of `N` elements, radius `R`, imaging an object inside a concentric circle of radius `r0`. Field components of angular order `n > kr0` decay as `(r0/ρ)^n` between the object and the ring (eq. 30), so the highest order that reaches the ring is `|l| < kr0` (eq. 31). Placing the first grating lobe outside the object then requires [S: Sim07]

`N > (2πr0/λ)(1 + 1/α_{N−1}) ≈ 4πr0/λ = 2k·r0` (eq. 32), i.e. pitch `Δ < λR/(2r0)` (eq. 33).

They state that eq. 33 reduces to the λ/2 pitch rule when `r0 = R`, "equivalent to the condition introduced by Lin et al. for circular scatterers" [S: Sim07, citing Lin00]. Two caveats are theirs: the derivation assumes `R` is "sufficiently larger than r0", and "when r0 approaches R ... aliasing can occur even if (33) is satisfied". A non-concentric object needs the radius of the concentric circle that encloses it [S: Sim07 §III]. The photoacoustic analogue is `n ≥ 2RΩ` detectors for a source of essential bandwidth Ω inside the detection circle [S: Hal16, as stated in Haltmeier et al. 2024]. The `2ka` degrees of freedom of a field scattered by a radius-`a` object are Bucci & Franceschetti's result [S: BF89 (abstract; the 2ka figure via secondary)].

**Which `r0` applies here** [I]:

- The head alone: `r0` = 99.8 mm. Then `2kr0` = 66.9 at 80 kHz.
- But `r0/R` = 0.89 here, which is Simonetti's "r0 approaches R" case. Orders above `kr0` decay only as `(0.89)^n`. Counting orders that still reach the ring above −20 dB (Hankel ratio, appendix item 5) gives `|l|` up to 42 at 80 kHz, so `N > 42 + kr0 = 75` [C].
- The inversion also updates `c` between the head and the ring, and puts its largest updates at the transducer cells (NSA §1, #111). The inverted unknown therefore extends to `r0 = R`, which gives `2kR` = 75.4 [C].

All three point to `2kR`. Required counts [C]:

| f | 2k·r0 (head) | 2kR (ring) | default 64 passes up to |
|---|---|---|---|
| 60 kHz | 50.2 | 56.5 | |
| 80 kHz | 66.9 | 75.4 | 76.5 kHz (head) / 67.9 kHz (ring) |
| 100 kHz | 83.6 | 94.2 | |

**Why it matters under the inverse crime** [I]. Simonetti's criterion is about the information on the ring, not about modelling error. Undersampled angular orders alias into lower ones in the data whether the data come from the true physics or from the inversion's own operator. With every transducer a source and a receiver, the multistatic data are sampled at the same pitch in both source and receiver angle, so the criterion applies to both. No source tests this for FWI or for a joint (ODIL) solve.

**For #138.** 76 transducers satisfy `2kR` at 80 kHz, and 96 at 100 kHz. #124's 100-transducer option (1.96× cost) over-samples 80 kHz.

## 4. Row 11: PML thickness (cross-check with #134)

- **No primary threshold applies to this layer.** The current layer damps only the mass term (`s_x s_y`) and leaves the Laplacian unstretched, so it is a graded damping layer with interface reflections, not a PML [R: #37, #134]. Its `sigma_max = −(p+1) c_max ln(R0) / (2L)` [R: `grid.py` `_sigma_max`] is the standard PML design formula; it predicts the reflection of a true PML, not of this layer [R: #37].
- **For a true PML,** thickness is set in cells, not wavelengths. Collino & Tsogka report about 1% reflection with a layer of five grid steps in heterogeneous anisotropic media [S: CT01 (abstract excerpt)]. A discrete PML's reflection depends on cells per layer and the grading profile; #134 owns that comparison.
- **The λ/2-at-`f_min` warning** in `wavefield/base.py` is a judgement call with no source found. At the default it is far from binding: 40 cells = 80.6 mm = 4.3λ at 80 kHz in 1500 m/s (3.2λ for #124's PML 30; #124's "3.7λ" doesn't match this arithmetic).
- **Severity** [I]. Under the inverse crime, the observations contain the same layer's reflections as the model, so they are not misattributed to `c`. They matter when `pml.observation_width` differs from `pml.width` (the existing deliberate modelling error [R: `config.py` `PmlCfg`]) and once observations are independent. #124 should move the runtime warning into `validate_config` as is, and #134 should replace the threshold.

## 5. Row 12: skull thin-layer detection

**The seismic criteria are reflection-mode, broadband and 1D.**

- **λ/8 (Widess).** Below about λ/8 the composite reflection from a thin bed stops changing shape: its amplitude scales with thickness, so only the product of thickness and reflection coefficient is constrained [S: Wid73 (via secondary; not read)]. It is a resolution limit; detection goes thinner and is limited by noise (same secondary sources).
- **λ/4 (Kallweit & Wood).** "The practical limit of resolution ... occurs at a one-quarter wavelength condition and approximates 1/(1.4 f_u)." They separate this from detection: the paper "deals with resolving reflections from thin beds rather than the detection of events" [S: KW82 abstract].

**For a full ring at one frequency** [I]. The model update is band-limited to scattering wavenumbers `|K| ≤ 2k` (VO09 eq. 29–30, as read in NSA §1). The point-spread function of a uniformly filled disc of radius `2k` is `2J₁(2kr)/(2kr)`, with its first null at `r` = 3.83/(2k) = 0.305λ and FWHM 0.35λ. So two surfaces resolve at `T ≳ 0.3λ`, between the λ/4 and λ/2 rules. Thickness and speed separately need the layer's profile, not just its integral. Below resolution only the integral is constrained (in transmission, the delay `T(1/c_w − 1/c_s)`), so I take ≳ λ/2 for "thickness plus speed". Which λ to use is open: in a Born picture around water it is λ_water; around the true skull it is the local λ_skull (more conservative).

**Frequencies needed for `T` = 4 mm** [I]:

| criterion | λ_skull (3000 m/s) | λ_water (1500 m/s) |
|---|---|---|
| λ/8: detect, trade-off `T`·contrast | 94 kHz | 47 kHz |
| λ/4: separate surfaces (reflection) | 188 kHz | 94 kHz |
| 0.305λ: separate surfaces (full-ring point-spread) | 229 kHz | 114 kHz |
| λ/2: thickness and speed | 375 kHz | 188 kHz |

At 80 kHz the rim is λ_skull/9.4 = λ_water/4.7. It is still clearly visible in transmission: the extra delay is 0.67 rad per crossing [I]. A grid point too: `T` is 2 cells at the default `h`, so the model can't represent a thickness profile anyway [R].

**Severity.** This depends on the truth (`T`, `c_skull`) and the goal, not on the discretisation. It belongs in an info-level "resolution report" for skull profiles (#124's own suggestion), not in the warnings [I].

## 6. Severity for every row

Proposed vocabulary [I]: **error** = the run is invalid, refuse it; **warn** = recorded in `config_warnings` on every run; **info** = printed by `--dry-run` and recorded in `metadata.json`, not a warning. "Inverse crime" = `pml.observation_width` is null, the only current switch [R: `config.py`]. Rows that matter only beyond it can key off that field now and off #139's observation path later.

| # | Constraint | Threshold basis | Inverse crime | Independent / `observation_width` set |
|---|---|---|---|---|
| 1–3 | Source width, low-pass, ring clearance | existing; not researched here | warn | warn |
| 4 | PPW ≥ 2π at `c_min` | implied by 1+2 | drop | drop |
| 5 | Dispersion | §2; ε judgement | info | warn |
| 6 | Ring sampling | §3 [S: Sim07] | warn | warn |
| 7 | FFT bin snap | judgement | info | info (warn if snap > a few % of f, judgement) |
| 8 | Temporal Nyquist `f < 1/(2dt)` | `λ_tt` is periodic in ω with period 2π/dt, so frequencies alias [I from `utils.py` formula] | error | error |
| 9 | Time symbol `(ωdt)²/24` | same symbol analysis as §2 [C]; default 0.16% = 2.3 m/s | info | warn, same tolerance as row 5; #116 |
| 10 | CFL | not a stability limit for a frequency-domain solve [R: #124] | drop | drop |
| 11 | PML thickness | §4; judgement until #134 | info | warn |
| 12 | Skull thin layer | §5 | info (resolution report) | info |
| 13 | Source spectrum at band frequency | judgement | info | info (#28 owns the scale) |

## 7. Open questions

- **ε for row 5**, and whether the tolerance should be phase (π/4 over 2R) or bias (m/s at `c_min`). The maintainer should pick; an ADR on severity policy (#129 "Not yet specified") could hold it.
- **Ring sampling under ODIL**: is the 64-transducer aliasing visible in an inverse-crime run? A one-key test (64 vs 80 transducers, same budget) would show whether #111's ring-adjacent artefacts change. NSA §5 proposes the same check.
- **Which λ for the thin layer** (water or skull) in a nonlinear joint inversion; no source found.
- **PML threshold**: replace the λ/2 rule once #134 and #37 measure reflection against width.

## 8. Unverified and limits

- **Not read in full:** Widess 1973 (λ/8 known only from secondary summaries; Crossref has no abstract); Kallweit & Wood 1982 (abstract only); Bayliss et al. 1985 (abstract only); Collino & Tsogka 2001 (search excerpt of the abstract; HAL blocked); Bucci & Franceschetti 1989 (abstract; the `2ka` figure only via secondary sources); Haltmeier 2016 (its condition only as restated by the same author in 2024; the Innsbruck preprint host didn't resolve). VO09 eq. 29–30 come from NSA's reading; the cycle-skipping π criterion is cited from memory of VO09, not re-read.
- **Read:** Kreiss & Ortiz 2002 §4 and Table 1; Alford, Kelly & Boore 1974 abstract and conclusions; Simonetti, Huang & Duric 2007 §III–IV in full.
- Every dispersion number is a plane-wave, homogeneous-medium analysis: no reflection at interfaces, no heterogeneity, no PML.
- The −20 dB order count in §3 assumes every order up to the cut-off is equally excited at `r0`, so it is conservative.
- Nothing was run through the solver; the script uses only the stencil coefficients and closed forms.

## Sources

Retrieved 2026-11-19.

- [KO02] Kreiss H.-O., Ortiz O.E. "Some mathematical and numerical questions connected with first and second order time dependent systems of partial differential equations." *Lect. Notes Phys.* 604:359 (2002). Read §4, Table 1. https://arxiv.org/abs/gr-qc/0106085
- [AKB74] Alford R.M., Kelly K.R., Boore D.M. "Accuracy of finite-difference modeling of the acoustic wave equation." *Geophysics* 39(6):834–842 (1974). Read abstract and conclusions. https://doi.org/10.1190/1.1440470, PDF: https://w.daveboore.com/pubs_online/alford_kelly_boore_geophysics_1974.pdf
- [BGT85] Bayliss A., Goldstein C.I., Turkel E. "On accuracy conditions for the numerical computation of waves." *J. Comput. Phys.* 59(3):396–404 (1985). Abstract. https://doi.org/10.1016/0021-9991(85)90119-6
- [Sim07] Simonetti F., Huang L., Duric N. "On the spatial sampling of wave fields with circular ring apertures." *J. Appl. Phys.* 101:083103 (2007). Read §III (eq. 13, 20–33) and §IV. https://doi.org/10.1063/1.2717086, PDF: https://sites.rochester.edu/ultrasound-tomography/wp-content/uploads/2022/11/On-the-spatial-sampling-of-wave-fields-with-circular-ring-apertures.pdf
- [Lin00] Lin F., Nachman A.I., Waag R.C. *J. Acoust. Soc. Am.* 108:899 (2000). Via Sim07 only.
- [Hal16] Haltmeier M. "Sampling conditions for the circular Radon transform." *IEEE Trans. Image Process.* 25(6):2910–2919 (2016). Via Haltmeier M., Ye M., Felbermayer K., Hinterleitner F., Burgholzer P., "Design, implementation and analysis of a compressed sensing photoacoustic projection imaging system," arXiv:2402.15750 (2024), §2: "n ≥ 2Rλ equally spaced transducers ... are required to stably recover any PA source u that has maximal essential wavelength λ; see [22]" (λ there is a bandwidth). https://arxiv.org/abs/2402.15750
- [BF89] Bucci O.M., Franceschetti G. "On the degrees of freedom of scattered fields." *IEEE Trans. Antennas Propag.* 37(7):918–926 (1989). Abstract only.
- [Wid73] Widess M.B. "How thin is a thin bed?" *Geophysics* 38(6):1176–1180 (1973). https://doi.org/10.1190/1.1440403. Not read; via secondary summaries.
- [KW82] Kallweit R.S., Wood L.C. "The limits of resolution of zero-phase wavelets." *Geophysics* 47(7):1035–1046 (1982). Abstract (Crossref). https://doi.org/10.1190/1.1441367
- [CT01] Collino F., Tsogka C. "Application of the perfectly matched absorbing layer model to the linear elastodynamic problem in anisotropic heterogeneous media." *Geophysics* 66(1):294–307 (2001). Abstract excerpt. https://doi.org/10.1190/1.1444908
- [VO09] Virieux J., Operto S. "An overview of full-waveform inversion in exploration geophysics." *Geophysics* 74(6):WCC1–WCC26 (2009). Via NSA.
- Repo, at `01d55fe`: `odil_wave/operator/spatial.py` (`FD2_COEFFS`), `odil_wave/operator/utils.py` (`λ_tt`), `odil_wave/grid/grid.py` (`_sigma_max`, `dt`), `odil_wave/wavefield/base.py` (PML warning), `odil_wave/experiment/config.py` (defaults, `PmlCfg.observation_width`), `odil_wave/models/velocity_models.py`.

## Appendix: dispersion and ring-sampling script

Standalone (numpy, scipy); run with `uv run python dispersion.py`. The output reproduces every [C] number above.

```python
"""Phase error of the repo's FD Helmholtz symbol (#135). Standalone: numpy + scipy."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import hankel1

FD2 = {  # copy of odil_wave/operator/spatial.py FD2_COEFFS
    2: (-2.0, (1.0,)),
    4: (-30 / 12, (16 / 12, -1 / 12)),
    6: (-49 / 18, (3 / 2, -3 / 20, 1 / 90)),
    8: (-205 / 72, (8 / 5, -1 / 5, 8 / 315, -1 / 560)),
    10: (-5269 / 1800, (5 / 3, -5 / 21, 5 / 126, -5 / 1008, 1 / 3150)),
}


def s1(order, th):
    """-h^2 times the symbol of the 1D d2/dx2 stencil at theta = k h (exact: theta^2)."""
    c0, cs = FD2[order]
    return -(c0 + 2 * sum(c * np.cos((m + 1) * th) for m, c in enumerate(cs)))


def rel_err(order, G, phi=0.0, wdt=0.0):
    """kappa/k - 1: numerical over exact wavenumber of a plane wave at G points per
    wavelength, direction phi, with the discrete time symbol at omega*dt = wdt
    (omega_eff = (2/dt) sin(omega dt / 2); wdt = 0 is the exact time symbol)."""
    th = 2 * np.pi / G
    target = th**2 * (np.sinc(wdt / 2 / np.pi) ** 2 if wdt else 1.0)

    def f(kap):
        return s1(order, kap * np.cos(phi)) + s1(order, kap * np.sin(phi)) - target

    return brentq(f, 0.5 * th, min(2.0 * th, np.pi * 0.999)) / th - 1.0


def worst(order, G, wdt=0.0):
    """rel_err in the worst direction (the cross stencil is symmetric about pi/4)."""
    return max((rel_err(order, G, p, wdt) for p in np.linspace(0, np.pi / 4, 19)), key=abs)


def ppw_needed(order, n_wl, eps, wdt=0.0):
    """Smallest G with |accumulated phase error| <= eps rad over n_wl wavelengths."""
    def g(G):
        try:
            return 2 * np.pi * n_wl * abs(worst(order, G, wdt)) - eps
        except ValueError:  # no propagating root: below the stencil's cutoff
            return np.inf

    lo = next(G for G in np.arange(2.5, 400, 0.5) if g(G + 0.5) <= 0)
    return brentq(g, lo, lo + 0.5)


if __name__ == "__main__":
    c_w, f, h, dt, R, r0 = 1500.0, 80e3, 0.25 / 124, 500e-6 / 1299, 0.1125, 0.0998
    lam = c_w / f
    G0, wdt0, N0 = lam / h, 2 * np.pi * f * dt, 2 * R / lam
    print(f"default: G={G0:.2f} ppw, omega*dt={wdt0:.4f}, path 2R={N0:.2f} wavelengths")
    print("\n1) phase-velocity error at the default grid:")
    for o in FD2:
        e_ax, e_dg = rel_err(o, G0), rel_err(o, G0, np.pi / 4)
        e_t = worst(o, G0, wdt0)
        print(
            f"  o{o:<2} axis {e_ax:+.2e} diag {e_dg:+.2e} worst-with-dt {e_t:+.2e}"
            f"  phase over 2R {2 * np.pi * N0 * abs(e_ax):.3f} rad"
            f"  c-bias {abs(e_ax) * c_w:.2f} m/s"
        )
    print("\n2) PPW needed (spatial only, worst direction):")
    for eps, name in [(np.pi / 4, "pi/4"), (0.1, "0.1 rad")]:
        for n_wl in [5, 10, 12, 15, 20, 30, 50]:
            row = "  ".join(f"o{o}:{ppw_needed(o, n_wl, eps):5.1f}" for o in FD2)
            print(f"  eps={name:7} D/lam={n_wl:3}: {row}")
    print("\n3) default ring path 2R, spatial only:")
    for ff in (80e3, 100e3):
        n = 2 * R / (c_w / ff)
        for eps in (np.pi / 4, 0.1):
            G = {o: ppw_needed(o, n, eps) for o in FD2}
            print(
                f"  f={ff / 1e3:.0f} kHz D/lam={n:.1f} eps={eps:.3f}: "
                + "  ".join(f"o{o}:{G[o]:.1f}({c_w / ff / G[o] * 1e3:.2f}mm)" for o in FD2)
            )
    print(f"\n4) time symbol alone: kappa/k-1 = {np.sinc(wdt0 / 2 / np.pi) - 1:+.2e}")
    print("\n5) ring sampling, Simonetti 2007 eq. 32 N > 2 k r0:")
    for ff in (60e3, 80e3, 100e3):
        k = 2 * np.pi * ff / c_w
        n_ev = next(
            n for n in range(1, 400)
            if n > k * r0
            and abs(hankel1(n, k * R) / hankel1(n, k * r0)) * np.sqrt(R / r0) < 0.1
        )
        print(
            f"  f={ff / 1e3:.0f} kHz: 2k*r0={2 * k * r0:.1f}  2kR={2 * k * R:.1f}"
            f"  highest order reaching ring above -20 dB: {n_ev} -> N > {n_ev + k * r0:.0f}"
        )
```

Output (excerpt; full tables in §2.2 and §3):

```
default: G=9.30 ppw, omega*dt=0.1935, path 2R=12.00 wavelengths
  o2  axis +2.01e-02 diag +9.76e-03 worst-with-dt +1.84e-02  phase over 2R 1.513 rad  c-bias 30.10 m/s
  o4  axis +1.12e-03 diag +2.84e-04 worst-with-dt -1.28e-03  phase over 2R 0.084 rad  c-bias 1.68 m/s
4) time symbol alone: kappa/k-1 = -1.56e-03
  f=80 kHz: 2k*r0=66.9  2kR=75.4  highest order reaching ring above -20 dB: 42 -> N > 75
```
