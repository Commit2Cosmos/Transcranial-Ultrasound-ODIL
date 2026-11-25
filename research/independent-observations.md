# Independent observations: how USCT and transcranial FWI avoid the inverse crime

Research for #136 (map #129). It is the spec for #139. Throwaway branch; never merged.

Question: how do ultrasound-tomography (USCT) and transcranial FWI studies generate observed data that the inversion's own discretisation did not produce? Which way is cheapest and sound for this repo, and what else changes once the data are independent (metrics, #46, #116, #28)?

Domain terms follow `GLOSSARY.md`: **Inverse crime**, **True model**, **Transducer**, **Shot**, **Trace**, **Offset**, **Interior**, **PML**, **PDE term**, **Data term**, **Trace weighting**, **Band**, **Run config**, **Default config**.

Every claim carries one provenance tag:

- **[S]**: established by a source. The key points to Sources. "(abstract)" means I reached only the abstract or a search excerpt.
- **[R]**: read from this repo's code or issues.
- **[P]**: measured in a pilot for this note (§5). Default config, one 80 kHz band, float64.
- **[I]**: my own inference or arithmetic.

Default config used throughout [R: `odil_wave/experiment/config.py`]: a 125² interior over 0.25 m (h = 2.016 mm), PML 40 cells, `space_order` 2, 64 transducers, a tone burst at 80 kHz with one 80 kHz band, Shepp-Logan truth, `t_max` = 500 µs, `init_nt` = 1300. That is 9.0 points per wavelength in 1450 m/s tissue and 9.3 in water [I].

## 1. Conclusion

1. **The field's standard is a different discretisation of the same physics, usually on a finer grid, plus additive white Gaussian noise.**
   - Forte generates data with k-Wave (time domain) at 0.2 mm and inverts with a 9-point frequency-domain Helmholtz scheme at ≥ 0.32 mm. He calls this "without inverse crime" even though the physics (2D acoustics) is the same [S: Forte25 §III-B].
   - WISE uses a first-order k-space scheme at 0.25 mm for data and a second-order one at 0.5 mm for inversion "to avoid inverse crime", citing Colton & Kress [S: WISE15 §IV-C3, §IV-D1].
   - Ali et al. ship k-Wave data at 0.15 mm and invert with a Helmholtz solver at 0.3 mm [S: Ali24 code].
   - Stronger mismatch, such as 3D data inverted in 2D, elastic or visco-acoustic data inverted acoustically, or variable density inverted for sound speed only, is a further step that some studies take [S: Forte25 §III-C; Guasch20 Methods; BrendersPratt07 (abstract); Marty21 (abstract)].
   - Several respected studies commit the inverse crime and say so [S: Lucka22 §4; Lozenski24 §IV]. So does the Stride tutorial [S: Stride `breast2D` scripts].
2. **Noise is almost always additive, white, zero-mean Gaussian, i.i.d. across channels, with one σ for every channel.** The levels used are:
   - SNR 30 dB and 20 dB [S: Lozenski24 §IV].
   - SNR 30 to 5 dB, with σ = RMS(f)·10^(−SNR/20) [S: Lucka22 App. F, eq. 25].
   - σ = 5% of a reference peak [S: WISE15 §IV-C5].
   - A noise floor matched to the device's power spectral density [S: Forte25 §III].

   Measured transcranial SNR is 39 dB in vivo (±10 dB with orientation) and 34 dB ex vivo [S: Guasch20 Results].
3. **For this repo, the cheapest sound path is the existing sparse Helmholtz solve on a grid refined 3× with `space_order` 4 and exact `iω`/`−ω²` symbols.** Run this way, the observation solve differs from the inversion in grid, stencil and time symbols [R, I].
   - It reuses `HelmholtzSolver`, which needs only a refined `Grid`. Because r = 3 is an integer, every inversion node is a fine node, so transducers and the interior map exactly [R, I].
   - It costs about 20 s per 80 kHz band with 64 shots on CPU, and about 2.5 GB of LU fill [P].
   - Its numerical error is about 1% against an r = 3 → r = 2 extrapolation [P, I].
   - A time-domain solver, or k-Wave through `k-wave-python`, is the more conventional route but more work. Keep it as a later cross-check (§4).
4. **The pilot's main finding: once data are independent, the inversion's own stencil is the dominant forward-model error.**
   - The default `space_order` 2 at 9 points per wavelength puts the inverse-crime data 47% (rel ℓ2 over traces) away from converged data for the *same* velocity model. `space_order` 4 is 3.7% away and `space_order` 8 is 1.2% [P].
   - The leapfrog time symbols (#116) alone are 5.9% [P].
   - At the default stencil, the modelling error is therefore far larger than any realistic noise: 30 dB SNR is 3.2% [I]. #139 should land opt-in, and the default stencil needs its own ticket (§7).
5. **How the true model is resampled matters as much as the solver.**
   - The pilot compared a Shepp-Logan truth built natively on the 3× grid with the coarse truth copied to the fine grid by nearest-neighbour. Their data differ by 26% [P].
   - So #139 needs two observation-truth modes, `inversion` and `native` (§6.1), and a rule for which truth the metrics score against.

### Recommendation (spec for #139)

- **Data-generation path:** a new `observation` config block replaces `pml.observation_width` (§6.1).
  - `refine` = 3, `space_order` = 4, `time_symbols` = `exact`.
  - The PML has the same physical thickness by default, or is thicker when chosen.
  - `truth` = `inversion` (piecewise-constant prolongation of the inversion-grid truth) or `native` (the phantom rebuilt on the fine grid).
  - Traces are sampled at coincident nodes.
- **Noise model:** circular complex Gaussian noise, added to the frequency-domain traces (§6.2).
  - It is i.i.d. over shots and receivers.
  - Each frequency has one σ_f = RMS_f(d_clean)·10^(−SNR/20).
  - It is seeded per (seed, FFT bin), so a frequency shared by two bands gets the same realisation.
  - Levels: noiseless, 40, 30 and 20 dB.
- **Default:** stays the inverse crime (`observation.mode: inverse_crime`) until the stencil ticket (§7) lands, because at `space_order` 2 independent data mostly measure dispersion [P, I].
- **Knock-on changes (§6.3):**
  - Truth-residual and noise-floor reference rows.
  - A `rel_wavefield_error` against the restricted fine field.
  - An inverse-crime flag in run metadata.
  - `balance.py` updated.
  - A `per_frequency` trace-weighting arm in #46.
  - #116 done next.
  - #28 does not block synthetic independent data.

---

## 2. What the literature does

### 2.1 Data-generation practice

| Study | Data generated with | Inverted with | Difference | Noise | Tag |
|---|---|---|---|---|---|
| Forte 2025, ring USCT, frequency-domain inversion | k-Wave 2D time domain, 0.2 mm pixels, FFT at bins (100–1000 kHz). Also k-Wave 3D with arc-shaped elements. | 9-point FD Helmholtz, pixels 0.8 → 0.32 mm by band | Solver (TD vs FD), grid, pixel size; 3D→2D in the "model mismatch" case. The phantom is interpolated to each grid. | Gaussian, scaled so that only > 300–400 kHz sits above the floor (system PSD); 14-bit ADC clipping | [S: Forte25 §III, §III-A–C, Table I] |
| Forte 2025, inverse-crime baseline | FD2D Helmholtz, 0.8 mm | Same grid and scheme | None (labelled "Inverse Crime") | None | [S: Forte25 §III-A] |
| WISE (Wang et al. 2015), ring USCT | First-order k-space PSTD, 0.25 mm grid, 20 MHz, decimated to 10 MHz | Second-order k-space, 0.5 mm grid | Scheme order and grid; time decimation "to avoid inverse crime [Colton & Kress]" | White Gaussian, σ = 5% of the peak of a reference water-shot trace, one σ for all channels | [S: WISE15 §IV-C1–C5, §IV-D1] |
| Ali et al. 2024 (open code), ring USCT | k-Wave 2D, dx = 0.15 mm; then a windowed DTFT at the inversion frequencies | Helmholtz (block LU), dx = 0.3 mm, point source on a grid node | Solver and 2× grid. Per-transmitter complex source scaling is estimated every iteration. | None added in the simulation script | [S: Ali24 code `GenKWaveSimInfo.m` L9, `MultiFrequencyWaveformInvkWave.m` L26–55, L60, L194–202] |
| Guasch et al. 2020, transcranial 3D | 3D variable-density acoustic TD FD, 10th order in space and 4th in time. Visco-acoustic for Supp. Fig. 2. | Sound speed only (density, absorption not inverted) | Physics: density and absorption are present in the data but absent from the inverted parameters. No grid difference stated. | Discussed in the Supplementary Information (not read). Lab SNR measured: 39 dB in vivo, 34 dB ex vivo. FWI "robust at signal-to-noise ratios below 2 dB". | [S: Guasch20 Methods, Results] |
| Lucka et al. 2022, 3D breast | k-Wave, same grid as the inversion | Same | None. Explicitly a validation of the computation. | None in the main study. App. F: Gaussian, σ = RMS(f)·10^(−SNR/20), SNR 30…5 dB. 30 dB is "visually indistinguishable" from clean. | [S: Lucka22 §4, App. F] |
| Lozenski et al. 2024, learned FWI | Devito FD | Same Devito discretisation | None; the paper names the "absence of modeling error (discretization inverse crime)" as a limitation-mitigator | White Gaussian, σ = 4.3e-6 = SNR 30 dB; 20 dB in study 3 | [S: Lozenski24 §IV] |
| Stride `breast2D` example | Devito, 0.5 mm, (356, 385) | Same grid | None | None | [S: Stride `01_script_forward.py`, `02_script_inverse.py`] |
| Brenders & Pratt 2007, seismic blind test | Third-party elastic wave-equation data | 2D frequency-domain acoustic waveform tomography | Physics and code, blind | Not checked | [S: BrendersPratt07 (abstract)] |
| Marty, Boehm & Fichtner 2021, transcranial | Spectral-element acoustoelastic | Acoustoelastic vs acoustic skull | Physics (elastic skull) | Not checked | [S: Marty21 (abstract)] |
| Robins et al. 2023; Mitcham et al. 2025 | Measured phantom / ex vivo data | — | Real data | Real | [S: Robins23; Mitcham25, via `research/fwi-metrics.md`] |

The definitions behind this practice:

- The inverse crime "occurs when the same (or very nearly the same) theoretical ingredients are employed to synthesize as well as to invert data", and Colton & Kress call it "trivial and therefore to be avoided" [S: Wirgin04 (abstract)].
- Kaipio & Somersalo treat the discretisation error as part of the noise model (the approximation-error approach) and illustrate the inverse crime [S: KaipioSomersalo07 (abstract)].

The consensus from the table [I]:

1. The minimal accepted remedy is **a different grid and a different scheme for the same PDE** (Forte, WISE, Ali).
2. The truth is defined or interpolated on the finer data grid (Forte, Ali).
3. Physics mismatch and measured data are later rungs.

### 2.2 Source and receiver model

- In 2D, transducers are point-like on a grid node in both data and inversion [S: Forte25 §III; WISE15 §IV-C1 (nearest-neighbour placement); Ali24 code L177–182].
- In 3D, Forte uses arc-shaped elements and Lucka idealised points [S: Forte25 §III; Lucka22 §4.1].
- When the data and the inversion do not share a source wavelet, the frequency-domain codes estimate one complex source factor per transmitter and frequency by least squares, `s = (u_simᴴ d)/(u_simᴴ u_sim)`, re-estimated every iteration [S: Ali24 code L201–202; Forte25 §III-B, "equation (17) in" Pratt99]. Forte notes this adds a parameter to the cost function.

### 2.3 Noise model and levels

- **Model:** additive, zero-mean Gaussian, i.i.d. across samples and channels, covariance σ²I [S: Lucka22 eq. 24; WISE15 eq. 24; Lozenski24 §IV]. WISE calls it "electronic measurement noise". Lucka says that whether a decorrelation transform makes the covariance close to a scaled identity awaits a real noise characterisation.
- **How σ is set:** SNR relative to the clean data's RMS [S: Lucka22 eq. 25], a fixed σ [S: Lozenski24], a fraction of a reference peak [S: WISE15], or matched to the device's noise-floor PSD [S: Forte25].
- **Levels:** 30 and 20 dB [S: Lozenski24]; 30, 25, 20, 15, 10 and 5 dB [S: Lucka22]; about 26 dB relative to the peak [I, from 5%: WISE15]. Measured transcranial SNR is 34–39 dB [S: Guasch20], defined in the time domain as the RMS in a 20 µs window after first arrival over the RMS before it.
- **Time vs frequency domain** [I]:
  - White time-domain noise of variance σ_t² becomes, after this repo's unnormalised DFT (`fft_norm=None` [R: `frequency_selection.py`]), circular complex Gaussian noise of variance nt·σ_t² in each bin. It is independent across bins.
  - A per-bin SNR therefore isn't the same number as a time-domain SNR. Defining the SNR directly on the frequency-domain traces avoids that mapping.

---

## 3. Where the repo stands [R]

- `Problem.make_band` always generates observed data with `HelmholtzSolver` on `observation_grid`. That is the inversion grid unless `pml.observation_width` ≠ `pml.width`, in which case a second grid with a different PML is built. It must share `nt` and `dt` [R: `problem.py` L83–117, L224–242].
- Even with the PML split, the observation grid has the same h, `space_order`, leapfrog `λ_t`/`λ_tt` symbols and truth pixels, so it is still an inverse crime in all but the absorbing layer [R: #116; `frequency_selection.py` L21–29, L98–99].
- Transducers are snapped to the nearest node of the grid they are built on [R: `acquisition_geometry.py::_place_ellipse`]. On a refined grid they would snap to different nodes, a geometry error, unless they are mapped from the inversion grid.
- The source density integrates to 1 on any grid: the Gaussian `exp(−r²/σ_s²)/(πσ_s²)` up to `4 exp(−π²σ_s²/h²)`, the point source `1/(dx dy)` [R: `acquisition_geometry.py::_spatial_profile`]. The spectrum is the unnormalised DFT of the waveform on `grid.t` [R: `source_signal.py::spectrum`]. It is identical on any two grids sharing `nt` and `dt`.
- The Shepp-Logan truth is a pixel model: skimage's phantom is resized to `round(n·scale)`, the skull rim is a 2-pixel binary erosion (`max(2, round(0.02·n))`), and `skull_sigma` is in grid cells [R: `velocity_models.py` L163–197, L261–296]. Rebuilt on a finer grid it is a different object (§5).
- Code that assumes the inverse crime:
  - `Band.true_wavefield_interior` and the tape's `rel_wavefield_error` assume the true wavefield lives on the inversion interior [R: `band.py` L87–96; `tape.py` L340–346].
  - `balance.py` evaluates the PDE term at `band.true_wavefields`, commenting "Inverse crime" [R: `balance.py` L115–121].
- There is no noise anywhere [R: grep].

---

## 4. Candidates for this repo

| Candidate | What differs from the inversion | New code | Cost per band (default, 80 kHz, 64 shots) | Verdict |
|---|---|---|---|---|
| **A. Refined-grid sparse Helmholtz** (r = 3, `space_order` 4, exact symbols) | Grid (h/3), stencil order, time symbols, optionally PML thickness and truth resolution | A `Grid` built from the inversion grid × r, transducers mapped `ij_fine = r·ij`, an exact-symbol option, restriction of traces and wavefields. `HelmholtzSolver` is unchanged. | 19 s solve; LU fill 1.56e8 entries (≈ 2.5 GB at complex128) [P] | **Recommended.** Matches the Forte/WISE/Ali rung (different grid and scheme, same PDE). Cheapest. |
| A′. Same at r = 2 or with `space_order` 2 | Fewer of the above | Same | 6 s (r = 2, o4); 5 s (r = 3, o2) [P] | Allowed by config, but r = 2 is 5% from r = 3 [P], so it is less converged. |
| A″. r = 3, `space_order` 8 | As A | Same | 104 s factorisation, LU 5.35e8 entries (≈ 8.6 GB) [P] | Too heavy for a default. |
| B. Own time-domain FD solver + FFT | Scheme (leapfrog TD vs Helmholtz), grid; transients and windowing | A new TD solver (one was removed from the repo [R: #116]), time windowing, a decay check against `t_max` | Not measured | Closer to Forte/Ali, but more code to own. The DFT of a truncated record equals the Helmholtz solution only if the response has decayed by `t_max` [I]. |
| C. k-Wave via `k-wave-python` or j-Wave | A third-party k-space PSTD scheme | Wrapper, file I/O, binaries (k-wave-python wraps the k-Wave GPU/CPU binaries; LGPL-3.0) [S: k-wave-python repo; jwave repo] | Not measured | The field's de facto reference [S: Forte25; Ali24; WISE15 used a k-space code]. Best as a one-off cross-check of A, not a dependency. |
| D. Physics mismatch (3D data, density, attenuation, elastic skull) | Physics | Large | — | Later rung. Out of scope for #139. |
| E. Measured data | Everything | Data loader, source estimation, calibration | — | No public 2D transcranial ring dataset at 80 kHz found. The public sets are breast USCT at 0.3–1.3 MHz [S: Ali24 README]. |

---

## 5. Pilot: how far apart are the data paths? [P]

Method:

- The default config at 80 kHz, float64.
- Traces from `HelmholtzSolver` on the default grid (r = 1) and on grids refined r× with the interior extent fixed: `(n−1)·r+1` nodes, PML `40·r` cells, same `nt`/`dt`.
- Transducers are mapped to `r·ij`, so their physical positions are identical. `skull_sigma` is scaled by r.
- "Exact" replaces `λ_t`, `λ_tt` with `iω_nd`, `−ω_nd²`.
- "NN truth" prolongs the inversion-grid truth to the fine grid by nearest neighbour, so the velocity model is the same. "Native truth" rebuilds the Shepp-Logan phantom on the fine grid.
- Numbers are ‖d_a − d_b‖/‖d_b‖ over all (shot, receiver) traces.

| Comparison | rel ℓ2 | What it isolates |
|---|---|---|
| Inverse-crime data (r1, o2, leapfrog) vs r1, o2, exact | 0.059 (median phase gap 0.085 rad) | Leapfrog vs exact time symbols (#116) |
| Inverse-crime data vs r3, o4, exact, NN truth | **0.470** | The default stencil's discretisation error, same model |
| r1, o4, exact vs r3, o4, exact, NN truth | 0.037 | A 4th-order inversion stencil |
| r1, o8, exact vs r3, o4, exact, NN truth | 0.012 | An 8th-order inversion stencil |
| r2, o4 vs r3, o4 (exact, NN truth) | 0.050 | Convergence of the observation path |
| r3, o4, exact: NN truth vs native truth | **0.264** | Truth resampling (skull rim and phantom edges at 0.67 mm vs 2 mm) |
| Inverse-crime data vs r3, o4, exact, native truth | 0.472 (median phase gap 0.61 rad; RMS amplitude ratio 0.947) | Both together |

Reading [I]:

- If the observation error scales as h⁴, the r = 2 → 3 gap of 5% means the r = 3, o4 data are about 1% from converged. That is consistent with r1/o8 sitting 1.2% from them.
- So 1% is the floor below which the "independent" data are themselves uncertain.
- At the default `space_order` 2, the inversion cannot fit converged data better than about 47% even at the true model. That is equivalent to an SNR of about 6.5 dB, worse than every noise level in §2.3. `space_order` 4 brings it to about 29 dB and 8 to about 38 dB.
- Rough dispersion arithmetic agrees. Second-order FD phase-velocity error ≈ (kh)²/24 ≈ 2% at 9 points per wavelength, which accumulates over about 11 wavelengths across the ring.

Factorisation benchmark (5-point and wider stencils, 80 kHz, 64 right-hand sides, complex128, SuperLU via `scipy.sparse.linalg.splu`):

| r | `space_order` | Grid nodes | LU nnz | Factor | Solve (64 RHS) |
|---|---|---|---|---|---|
| 1 | 2 | 205² | 3.7e6 | 0.2 s | 0.2 s |
| 2 | 2 | 409² | 1.9e7 | 1.1 s | 0.7 s |
| 3 | 2 | 613² | 4.7e7 | 2.9 s | 1.9 s |
| 3 | 4 | 613² | 1.6e8 | 15.4 s | 4.0 s |
| 3 | 8 | 613² | 5.4e8 | 103.6 s | 12.8 s |
| 4 | 4 | 817² | 2.9e8 | 49.9 s | 9.8 s |

Observations are generated once per band, so r = 3, o4 adds about 20 s per frequency to a run [P, I].

---

## 6. Spec for #139

### 6.1 Data-generation path

Replace `pml.observation_width` with an `observation` block. ADR-0001 applies: library defaults live in their owner's signature.

| Key | Values | Default | Meaning |
|---|---|---|---|
| `observation.mode` | `inverse_crime` \| `independent` | `inverse_crime` | `inverse_crime` keeps today's behaviour exactly |
| `observation.refine` | odd int ≥ 1 | 3 | Fine grid = `(n−1)·r+1` interior nodes over the same extent; same `nt`, `dt` (asserted) |
| `observation.space_order` | 2–10 | 4 | Stencil of the observation solve |
| `observation.time_symbols` | `exact` \| `leapfrog` | `exact` | `exact` = `iω`, `−ω²` |
| `observation.pml_thickness_factor` | float ≥ 1 | 1 | Physical PML thickness relative to the inversion's: `round(factor·pml.width·r)` fine cells. It subsumes `observation_width`. |
| `observation.truth` | `inversion` \| `native` | `inversion` | `inversion`: piecewise-constant (nearest-node) prolongation of the inversion-grid truth, so the true model is exactly representable and scoring is unchanged. `native`: the phantom rebuilt on the fine grid, with cell-unit knobs (`skull_sigma`) scaled by r. |
| `observation.noise.snr_db` | float \| null | null | null = noiseless (§6.2) |
| `observation.noise.seed` | int | 0 | Recorded; repeats vary it (#51) |

`refine` is restricted to odd values so the fine grid is also symmetric about the same centre. Any integer r makes every inversion node a fine node, `(i·r, j·r)` in full-grid indices because the PML scales by r too [I].

Behaviour:

1. Build the fine `Grid` from the inversion grid. Build the transducer ring, then **set `transducer_ij = r · transducer_ij_inversion`** rather than re-snapping. This keeps positions identical, as in the pilot [R, I].
2. Use the same `SourceSignal` parameters. Keep `source_spatial: gaussian`: its density is resolution-independent. A `point` source is `1/(dx dy)` at one node, a different physical source on each grid [R, I].
3. Solve, then extract traces at the receivers. Store `true_wavefields` restricted to coincident nodes (`u_fine[..., ::r, ::r]` over the interior, with `r·pml` offset), so they live on the inversion grid's interior [I].
4. Add noise (§6.2) to the traces only.
5. Record the observation settings and an `inverse_crime: bool` in run metadata, as `research/fwi-metrics.md` recommended.

Acceptance criteria to carry into #139:

- `mode: inverse_crime` reproduces today's `d_obs` bit-for-bit, and the shipped `current` records are unaffected.
- With `refine: 1`, `space_order` equal to the inversion's, `time_symbols: leapfrog` and `truth: inversion`, `independent` mode also reproduces `d_obs`. This guards the plumbing.
- A test that transducer physical positions match between grids.
- A test that the r = 3 vs r = 2 observation gap is below a pinned tolerance (the pilot gives 5% at the default config; float32 may differ).
- A test that a homogeneous truth gives data within about 1% of a converged reference [I].
- Observation cost is logged.

### 6.2 Noise model

- `d_noisy[s, f, k] = d_clean[s, f, k] + σ_f · (n₁ + i n₂)/√2`, with n₁, n₂ ~ N(0, 1) i.i.d. over shots s, receivers k and frequencies f [I; the Gaussian i.i.d. model follows Lucka22, WISE15 and Lozenski24].
- `σ_f = RMS_{s,k}(|d_clean[s, f, k]|) · 10^(−SNR/20)`. This is Lucka's eq. 25 applied per frequency to the frequency-domain traces [S: Lucka22; I for the per-frequency choice]. One σ per frequency is the "same σ on every channel" of all the cited studies.
  - Per frequency rather than global, because the source spectrum differs between bands and a global σ would make the SNR depend on band placement [I].
  - A physical white floor would give a constant σ across bins. Forte's PSD-matched floor is that option [S: Forte25]. Offer it later as `noise.reference: global` if needed.
- Seed with `(seed, fft_bin)`, so a frequency in two bands gets the same realisation, and a run is reproducible [I].
- Levels for experiments: noiseless, 40, 30 and 20 dB.
  - 40 dB brackets the measured 34–39 dB in vivo [S: Guasch20].
  - 30 and 20 dB are the learned-FWI and Lucka levels [S: Lozenski24; Lucka22].
  - Below 20 dB, nothing in this repo's questions needs it yet [I].
- These SNRs are per frequency bin, not the time-domain SNR of the cited studies (§2.3). Say so in records [I].

### 6.3 Knock-on changes

**Metrics (#42 / tape):**

- Add per band:
  - `rel_data_residual_truth`: the inversion's forward model at the true model, ‖P H_inv(c*)⁻¹f − d_obs‖/‖d_obs‖. It is the modelling-error-plus-noise floor (47% today at the default stencil [P]).
  - `noise_floor` = 10^(−SNR/20).
  - The final `rel_data_residual` is read against both. Fitting below the floor is overfitting noise (the discrepancy principle) [I].
- `rel_wavefield_error` uses the restricted fine wavefield. It now includes the fine-vs-coarse discretisation gap, so it is not 0 at the truth [I].
- `pde_rel_residual` of the true wavefield is no longer 0; report it at the start row [I].
- Model metrics: with `truth: inversion` they are unchanged. With `truth: native`, score against the inversion-grid build of the same phantom (today's `truth_velocity`), and state that this truth didn't generate the data. The pilot shows that choice alone moves the data by 26% [P].

**`balance.py`:** evaluate the PDE term at the inversion-grid solve `H_inv(c*)⁻¹f`, not at `band.true_wavefields` [R, I].

**#46 trace weighting:**

- Under §6.2's noise, σ is constant over a frequency's traces. The maximum-likelihood weighting is therefore uniform within a frequency, scaled by 1/σ_f across frequencies. Call it `per_frequency`: divide every trace of frequency f by the RMS of `|d_obs|` over all its shots and receivers [I].
- `per_shot` equals it only to the extent that each shot's RMS is the same. On the centred ring with a near-centred phantom they should be close; measure the spread [I].
- `per_trace` and `per_offset` up-weight low-amplitude far-offset traces relative to their noise, as #46's own limitation section says [R: #46].
- Rerun #46 with noise, and add `per_frequency` as an arm.

**#116 exact symbols:**

- Unblocked by `time_symbols: exact`. Against exact-symbol data, the leapfrog symbols are a 5.9% trace error with a median phase gap of 0.085 rad at the default config [P].
- That exceeds the `space_order` 8 spatial error (1.2%) and is comparable to `space_order` 4's (3.7%) [P]. So #116 matters as soon as the stencil is raised. Schedule it right after #139.

**#28 source scale:**

- Not a blocker for synthetic independent data. Observation and inversion share `nt`, `dt`, waveform and a unit-integral source density, so the DFT scale cancels [R, I].
- It matters for a time-domain or k-Wave path (candidates B and C), and for measured data. There the field estimates one complex factor per shot and frequency each iteration [S: Ali24 code; Forte25; Pratt99]. In the joint objective it could be either a new joint unknown or a closed-form least-squares estimate from the current receiver wavefield. Neither eliminates `u`, so ADR-0003 (no variable projection or WRI for the wavefield) doesn't rule them out, but either changes the objective [R: `docs/adr/0003-no-variable-projection-or-wri.md`; I]. Revisit then.
- The snapped-bin labels (#28) don't affect this path, because both grids share bins [R].

**#51 (noise, seeds, repeats):** `noise.seed` and `noise.snr_db` make noise and repeats expressible. Experiment specs can sweep the seed [I].

**Experiment records:** `mode: inverse_crime` stays the default, so no record goes stale through #139 itself. Flipping the default later makes every record that relies on exact data stale (`docs/experiments/README.md`) [R].

---

## 7. New questions (candidate tickets)

1. **Default inversion stencil.** At 9 points per wavelength, `space_order` 2 is a 47% trace error against converged data; 4 is 3.7% and 8 is 1.2% [P]. What does a higher order cost in the joint ODIL objective? The stencil is matrix-free there, but `wavefield_preconditioner` LU fill grows, as §5 shows for SuperLU. Does order or h change the convergence of the joint solve? This decides when `observation.mode: independent` can become the default.
2. **Truth definition across resolutions.** The Shepp-Logan builder is pixel-defined (erosion in pixels, `skull_sigma` in cells, `resize` to `round(n·scale)`). A resolution-independent phantom, such as analytic ellipses with a skull of fixed thickness in metres, would make `truth: native` and scoring well defined.
3. **Cross-check with k-Wave or a time-domain solver.** Run one band of candidate A against `k-wave-python` or j-Wave to confirm the about 1% observation accuracy with an independent code. This also exercises #28's source estimation.
4. **Physics mismatch rung.** Density in the data only (as Guasch20), attenuation, or 3D→2D (as Forte25): which first, once the stencil is fixed?

---

## 8. Unverified and limits

- **Guasch20 Supplementary Information** (noise robustness, absorption) was not read. The "below 2 dB" claim is from the main text.
- **BrendersPratt07, Marty21, KaipioSomersalo07, Wirgin04, Pratt99:** abstract-level or via citing papers only. Pratt's source-estimation formula is read from Ali's code and Forte's description of it, not from Pratt99 itself.
- **Forte25** is an arXiv preprint (2025). Its "without inverse crime" case shares 2D acoustic physics with the inversion by design.
- **Pilot:** one frequency, one phantom, float64. The run config default is float32, where the bit-for-bit criteria may need a tolerance. The `space_order` 4 and 8 rows use exact symbols. Only the default `space_order` 2 was measured with leapfrog symbols. The pilot scripts are not committed; §5 describes them fully.
- **No source found** for frequency-domain noise added directly to complex traces in USCT. All cited studies add noise in the time domain. The §6.2 model is the frequency-domain image of i.i.d. white time-domain noise [I].

## Sources

- [Forte25] Forte L.A. "Stochastic Reconstruction of the Speed of Sound in Breast Ultrasound Computed Tomography with Phase Encoding in the Frequency Domain." arXiv:2508.08434 (2025). https://arxiv.org/abs/2508.08434
- [WISE15] Wang K., Matthews T., Anis F., Li C., Duric N., Anastasio M.A. "Waveform inversion with source encoding for breast sound speed reconstruction in ultrasound computed tomography." *IEEE TUFFC* 62(3):475–493 (2015). arXiv:1501.00021. https://arxiv.org/abs/1501.00021
- [Ali24] Ali R., Mitcham T.M., Brevett T., Calderón Agudo Ò., Durán Martinez C., Li C., Doyley M.M., Duric N. "2-D Slicewise Waveform Inversion of Sound Speed and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU Solver." *IEEE TMI* 43(8):2988–3000 (2024). Code: https://github.com/rehmanali1994/WaveformInversionUST (`Simulations/GenKWaveSimInfo.m`, `MultiFrequencyWaveformInvkWave.m`, README)
- [Guasch20] Guasch L., Calderón Agudo O., Tang M.-X., Nachev P., Warner M. "Full-waveform inversion imaging of the human brain." *npj Digit. Med.* 3, 28 (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7060331/ (full text read via Europe PMC)
- [Lucka22] Lucka F., Pérez-Liva M., Treeby B.E., Cox B.T. "High resolution 3D ultrasonic breast imaging by time-domain full waveform inversion." *Inverse Problems* 38(2):025008 (2022). https://arxiv.org/abs/2102.00755
- [Lozenski24] Lozenski L., Wang H., Li F., Anastasio M., Wohlberg B., Lin Y., Villa U. "Learned Full Waveform Inversion Incorporating Task Information for Ultrasound Computed Tomography." *IEEE TCI* (2024). https://arxiv.org/abs/2308.16290
- [Stride] Cueto C. et al. Stride, `stride_examples/examples/breast2D/01_script_forward.py`, `02_script_inverse.py`. https://github.com/trustimaging/stride
- [BrendersPratt07] Brenders A.J., Pratt R.G. "Full waveform tomography for lithospheric imaging: results from a blind test in a realistic crustal model." *GJI* 168(1):133–151 (2007). https://ouci.dntb.gov.ua/works/4Y3xAWP9 (abstract only)
- [Marty21] Marty P., Boehm C., Fichtner A. "Acoustoelastic full-waveform inversion for transcranial ultrasound computed tomography." *Proc. SPIE* 11602, 1160211 (2021). https://doi.org/10.1117/12.2581029 (abstract only)
- [Pratt99] Pratt R.G. "Seismic waveform inversion in the frequency domain, Part 1: Theory and verification in a physical scale model." *Geophysics* 64:888 (1999), as cited in Forte25 ref. [25]. Not read; cited via Forte25 and Ali24.
- [Wirgin04] Wirgin A. "The inverse crime." arXiv:math-ph/0401050 (2004). https://arxiv.org/abs/math-ph/0401050
- [KaipioSomersalo07] Kaipio J., Somersalo E. "Statistical inverse problems: Discretization, model reduction and inverse crimes." *J. Comput. Appl. Math.* 198(2):493–504 (2007). https://doi.org/10.1016/j.cam.2005.09.027 (abstract only)
- [Robins23], [Mitcham25]: see `research/fwi-metrics.md` Sources.
- [k-wave-python] https://github.com/waltsims/k-wave-python ("A Python interface to k-Wave GPU accelerated binaries", LGPL-3.0)
- [jwave] https://github.com/ucl-bug/jwave (JAX acoustic simulation, LGPL-3.0)
