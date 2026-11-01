# Reconstruction and data-fit metrics in ultrasound FWI

Research for #52 (map #51). It grounds the metrics review in #42. Throwaway branch; never merged.

Question: which metrics do ultrasound and transcranial FWI papers use to judge (a) sound-speed reconstruction quality and (b) data fit and convergence? How are they masked, normalised and reported, and which mislead?

Domain terms follow `GLOSSARY.md`: **Interior**, **PML**, **PDE term**, **Data term**, **Transducer**, **Shot**, **Offset**, **Inverse crime**.

## Summary

- Transcranial and USCT FWI papers report few image metrics. The flagship transcranial paper judges its reconstructions visually and reports no model-error number [Guasch20]. Where numbers appear they are **RMS error in m/s against the numerical brain model** [Robins23], **relative ℓ2** ‖c−c*‖/‖c*‖ [Lucka22], **region means ± std** from ROIs [Robins23, Mitcham25], **CNR/SNR** from ROI-vs-annulus [Mitcham25], and **SSIM + RMSE + a task score (lesion Dice)** [Lozenski24].
- Learned and ML-FWI benchmarks standardise on **MAE, (R)MSE, SSIM**, and the normalisation differs between papers [UPFWI22]. Numbers are not comparable unless the scale (m/s vs normalised) and SSIM settings are stated [skimage, Breger24].
- Data fit is almost always the **least-squares misfit ½‖d_syn − d_obs‖²**, in absolute units [Guasch20, Lucka22, Robins23, VirieuxOperto09]. Papers plot it against iterations or gradient evaluations [Lucka22] but rarely break it down by frequency or offset. Trace overlays are the usual data-fit evidence [Guasch20, Robins23].
- Joint wavefield-and-model methods (WRI / IR-WRI, the closest published relatives of joint ODIL) track **four** curves: model error, wavefield error, wave-equation (PDE) residual and data residual. They also plot the data-residual vs PDE-residual path [Aghamiry19]. ODIL and PINN papers report error against a reference, not the loss [Karnakov24].
- Known ways these metrics mislead:
  - **A good data fit does not mean a good model.** Close trace matches went with an 8.45 m/s brain error and an unrecovered skull [Robins23].
  - **Pixel metrics can disagree with each other and with the task.** FWI beat a learned method on RMSE but lost on SSIM and on lesion Dice [Lozenski24].
  - **SSIM pools uniformly, is unstable in low-variance regions and depends on its settings** [PambrunNoumeir15, Breger24, skimage].
  - **Inverse-crime synthetics flatter every metric** [Wirgin04, KaipioSomersalo07]. Several cited studies commit it or do not rule it out [Lozenski24, Lucka22].

---

## (a) Reconstruction-quality metrics

### Metrics found in the literature

| Metric | Definition as used | Where | Mask / ROI | Normalisation |
|---|---|---|---|---|
| Relative ℓ2 | ‖a−b‖₂/‖b‖₂, in %, b = ground truth c† | [Lucka22] §4.2 | Not stated explicitly. The reconstruction region Γ is "all pixels containing breast tissues" (Fig. 1). | By ‖c†‖, which includes the large background |
| Depth-resolved error | mean ± std of c_rec − c† per depth slice | [Lucka22] §4.2, Fig. 4 | Slices | m/s |
| RMS error | RMS(c_rec − c_true), m/s | [Robins23] (2.14 to 37.06 m/s, synthetic only) | "relative to the true numerical brain model". The mask is not defined. | Absolute m/s |
| Region mean ± std | Sound-speed mean and std over random samples or ROIs per tissue layer | [Robins23], [Mitcham25] (3 × 5 mm circular ROIs per feature) | Skull layer, brain layers, cavity; ventricles, grey/white matter, haemorrhage | m/s |
| Velocity histograms | Histogram of recovered c per region | [Robins23] Figs 3, 5 | Region | Not applicable |
| CNR, SNR | ROI vs an equal-area surrounding annulus (Eqs 1–2) | [Mitcham25] §2.2 | Lesion (haemorrhage) ROI + annulus | Dimensionless / dB |
| "Relative MSE" + SSIM | Ensemble mean over 41 test phantoms | [Lozenski24] §IV.D | Not stated | Not stated |
| Task score (Dice) | 2·N_true_det / (N_det + N_true_tumours), from a U-Net observer thresholded at 0.02 | [Lozenski24] §IV.D, Eq. 13 | Tumours | Dimensionless |
| MAE, MSE, SSIM | Pixel-wise errors + SSIM | [UPFWI22] (OpenFWI data) | Whole map | MAE/MSE on the denormalised (m/s) scale; SSIM on the [−1, 1] normalised scale |
| Resolution | Resolution lengths from the Hessian, as an alternative to checkerboard recovery tests | [FichtnerTrampert11]; [FichtnerVanLeeuwen15] say Hessian probing is "more meaningful than recovery tests" | Spatially varying | Length |
| Expected resolution limit | ≈ half the propagated wavelength | [VirieuxOperto09] abstract | Not applicable | Not applicable |
| Qualitative only | Visual agreement with the true model; structures down to 0.8 mm "recovered" | [Guasch20] Results | Anatomical regions by eye | Not applicable |

The low-frequency transcranial ring device paper reportedly scores phantom accuracy with SSIM and per-region precision (mean ± std). This came from a search snippet only; the full text returned 403 (see Unverified) [CudeiroBlanco22].

### What each detects, and how it misleads

- **Relative ℓ2 over the whole interior** [Lucka22]. It detects global error. It misleads because the denominator ‖c*‖ is dominated by the ~1500 m/s background (water/brain), so a homogeneous starting model already scores a few percent. Lucka's initial model scores 2.18%. A skull error or a small lesion barely moves it. Repo note: our analysis, not a cited finding.
- **RMSE in m/s** [Robins23]. Physically interpretable and comparable across grids if the mask is fixed. It misleads when the mask is unstated, as in Robins: brain-only and whole-object numbers differ by an order of magnitude when the skull is in or out. MSE-family metrics ignore the spatial structure of the error: very different distortions can share one MSE [WangBovik09, Breger24].
- **Region means ± std / histograms** [Robins23, Mitcham25]. They detect bias per tissue (e.g. the skull not updated from its starting model) and homogeneity. They mislead because hand-placed ROIs choose what gets measured, and the std mixes noise with real heterogeneity. Mitcham reports the std *across ROIs* as the error, not within them.
- **CNR / contrast recovery** [Mitcham25]. Detects lesion conspicuity. It misleads because it is relative, not quantitative: a biased lesion value can still give high CNR.
- **SSIM** [Wang04]. Detects structural/edge agreement. It misleads in four ways:
  - It pools uniformly, so local lesion errors average away [PambrunNoumeir15, Breger24].
  - It is unstable in low-variance regions, such as a homogeneous brain [PambrunNoumeir15].
  - It underestimates distortion near sharp edges [PambrunNoumeir15], which matters at the skull boundary.
  - It is sensitive to small misalignment [Breger24] and to parameters. Reproducing Wang 2004 needs `gaussian_weights=True, sigma=1.5, use_sample_covariance=False` and an explicit `data_range`. For float images skimage's default range estimate is wrong [skimage].
- **Task metrics (Dice, detection)** [Lozenski24]. Measure what matters clinically. "Metrics of task performance may not be directly correlated to physical metrics of image quality" (§II.C). They need a defined task and observer.
- **Visual agreement** [Guasch20]. Not reproducible and not comparable across sweeps.

### Masking practice

- Transcranial papers score the **brain** (intracranial) region separately from the skull. They report skull failure as a region-level bias ("did not update the skull starting model") rather than as a number [Robins23].
- Skull-start sensitivity is large. Brain RMS was 37.06 m/s from a water start, 8.45 m/s from a predicted skull and 3.35 m/s from the true (fitted) skull. Robins flags the last as unavailable clinically [Robins23]. So any brain metric must record what the skull prior was.
- Lesion metrics use small ROIs plus a background annulus [Mitcham25] or segmentations [Lozenski24].
- No cited paper specifies whether SSIM windows straddling a mask boundary are kept. That choice is implementation-defined (see the repo note in (d)).

## (b) Data-fit and convergence metrics

### Data misfit

- **Definition.** Standard FWI minimises ½‖Δd‖² per source, summed over sources [Lucka22 §2.3, Guasch20 Methods, Robins23]. In the general form ½Δd†W_dΔd, the diagonal weight S_d can, for example, be a power of offset "to strengthen the contribution of large-offset data" [VirieuxOperto09, eq. 17].
- **Normalisation.** Papers report the misfit in absolute units [Lucka22, Aghamiry19]. Source-independent alternatives normalise each trace by the sum over the shot gather [VirieuxOperto09]. AWI replaces differences with a per-trace ratio (a deconvolution towards a zero-lag delta) [Guasch20 Methods]. Absolute misfit is not comparable across bands, sources or noise levels. A relative misfit ‖d_syn − d_obs‖/‖d_obs‖ (cf. skimage `normalized_root_mse`, 'euclidean' [skimage]) is the natural fix. That is our suggestion; no cited USCT paper reports one.
- **Robust norms.** L2 assumes Gaussian residuals. L1 / Huber / Cauchy are less sensitive to outliers [VirieuxOperto09, after Crase et al. 1990].

### Residual by offset and frequency

- **Why offset matters.** Residuals within half a period of the first arrival back-project constructively over the first Fresnel zone, of width ~√(λ·offset). Beyond half a period the result is cycle skipping and a local minimum. The tolerance tightens with the number of propagated wavelengths, Δt/T < 1/N_λ [VirieuxOperto09]. The same half-cycle criterion drives transcranial practice: the skull delay of ~2.5 μs is about one 400 kHz cycle, so water-start data are cycle-skipped [Guasch20]. In a ring, the misfit at long **Offset** and high frequency is where cycle skipping shows up first. A residual-by-offset curve per band is therefore the direct diagnostic. That is our inference from these sources; no cited USCT paper plots one.
- **Why frequency matters.** Frequency-domain FWI inverts single frequencies or frequency groups from low to high. The frequencies are chosen so each frequency's maximum wavenumber meets the next one's minimum, so spacing grows with frequency [VirieuxOperto09, after Sirgue & Pratt 2004]. Transcranial schedules: 8 bands ~100–850 kHz with 10 iterations per band [Guasch20]; 22 bands 0.55–1.6 MHz with 3 iterations per band [Robins23]; 50 kHz steps from 300 or 50 kHz to 700 kHz [Mitcham25]. None of these report misfit per band.
- **Phase-residual QC.** The Imperial FWI group (Shah, Warner et al., SEG 2012) has a QC paper on starting-model adequacy, *Quality assured full-waveform inversion*. Its content (likely phase residuals) could not be verified (see Unverified).

### PDE / physics residuals for joint methods

- **WRI / IR-WRI** jointly solve for wavefield and model with J = ‖Pu−d‖² + λ‖A(m)u−b‖² [Aghamiry19, eq. 7]. This matches joint ODIL's PDE term plus data term. Their diagnostics, Fig. 2:
  - relative model error ‖m^k − m*‖/‖m*‖
  - relative wavefield error ‖u^k − u*‖/‖u*‖
  - wave-equation misfit ‖A(m^k)u^k − b‖
  - data misfit ‖Pu^k − d‖

  Fig. 3 plots the path in the (data misfit, wave-equation misfit) plane. The stopping rule needs **both** wave-equation misfit ≤ δ and data misfit ≤ ε (eq. 28). With small λ, large wave-equation error early on is expected and deliberate ("expanding the search space") [Aghamiry19].
- **Misleading mode.** A low total loss can be a low data term bought with a violated PDE (the wavefield fits the receivers but is not a wave), or the reverse. Weighting the terms decides which happens [Aghamiry19]. PINNs show the same imbalance as "unbalanced back-propagated gradients" between composite-loss terms [WangTengPerdikaris20], and as ill-conditioned loss landscapes [Krishnapriyan21].
- **ODIL papers** report the L2 error against a reference (relative to the *discrete* solution for forward problems) versus iterations and wall time. They report ODIL vs PINN as error vs number of parameters, epochs and time, with 25 seeds for PINN scatter. They do not use the loss value as the convergence metric [Karnakov24].

## (c) Reporting practice

- **Per iteration.** Misfit (or its stochastic estimate) vs number of gradient evaluations, rescaled to GPU-hours when comparing schemes [Lucka22 Fig. 6, Fig. 8]. Joint methods: model error, wavefield error, PDE residual and data residual vs iteration [Aghamiry19].
- **Per band / level.** Lucka marks level transitions on the error curves and notes "each level transition involves a smoothing which can lead to an increase in rel ℓ2 error" [Lucka22 §4.7]. So per-band boundaries must be visible on curves, and an error rise at a boundary is not divergence. Transcranial papers give the band schedule but no per-band numbers [Guasch20, Robins23, Mitcham25].
- **Final.**
  - Final model images, region means ± std, RMS or relative ℓ2 [Robins23, Lucka22, Mitcham25].
  - Total iteration counts [Aghamiry19].
  - For learned methods, ensemble means over a held-out test set [Lozenski24].
- **Comparing sweeps / methods.**
  - Equal-cost comparisons: error vs GPU-hours [Lucka22], or error-vs-time with a defined threshold ("150% of the error after 80000 epochs") [Karnakov24].
  - Seed scatter over repeated runs [Karnakov24].
  - More than one metric, since rankings flip between RMSE, SSIM and task score [Lozenski24].
  - Starting-model and skull-prior conditions stated alongside the numbers [Robins23].
- **Inverse crime.**
  - Generating and inverting data with "the same (or very nearly the same) theoretical ingredients" is "trivial and therefore to be avoided" [Wirgin04, after Colton & Kress].
  - Ignoring the discretisation error "may have serious consequences to the quality of the reconstruction" [KaipioSomersalo07].
  - Lucka partly mitigates by generating at 0.5 mm and inverting at 2 mm, but uses the same solver and no noise [Lucka22 §4.2]. Lozenski names "absence of modeling error (discretization inverse crime)" as a caveat [Lozenski24].
  - Metrics from inverse-crime runs should be labelled as such.

## (d) Mapping onto current repo metrics. *Suggestions for the #42 grilling, not decisions.*

Current state (read from code):
- `FIELDNAMES` is in `odil_wave/experiment/recorder.py`.
- Final recovery metrics come from `_final_recovery_metrics` in `odil_wave/experiment/runner.py`.
- Residuals come from `InverseLoss.evaluate` in `odil_wave/loss/inverse.py`.
- SSIM lives in `odil_wave/metrics/metrics.py`.

How the current metrics are computed:
- `data_rms` = √mean|syn − obs|² over all shots × frequencies × receivers, absolute.
- `pde_rms` is absolute.
- `pde_src_ratio` = pde_rms / src_rms.
- `rel_c_error` = ‖c−c*‖/‖c*‖ over the whole interior.
- `ssim_head_roi`: the per-pixel SSIM map is computed over the interior and averaged inside `head_mask`. Defaults are win 7, no Gaussian, skimage's default sample covariance; `data_range` is the truth's peak-to-peak inside the mask.

| Repo metric | Suggestion | Rationale (source) |
|---|---|---|
| `loss_total`, `pde_term`, `data_term`, `reg_term` | **Keep** | Optimiser bookkeeping. Joint methods need both terms tracked separately [Aghamiry19]. |
| `pde_loss`, `data_loss` (unweighted MSE) | **Keep or drop** (redundant with `*_rms` squared) | Same information as `pde_rms`/`data_rms`. |
| `data_rms` | **Change**: add a relative form ‖d_syn−d_obs‖/‖d_obs‖ per band | Absolute misfit is not comparable across bands or configs [Lucka22, Aghamiry19 use absolute; our inference]. skimage NRMSE 'euclidean' is this normalisation [skimage]. |
| `pde_rms` | **Keep as raw, prefer `pde_src_ratio` for comparison** | Absolute; depends on grid and source scale. |
| `pde_src_ratio` | **Keep** (it is the normalised PDE residual; ‖Au−b‖/‖b‖-like) | Matches the wave-equation-misfit diagnostic of joint methods [Aghamiry19]. |
| (none) | **Add**: data residual **per frequency** and **per Offset bin** at band end | Cycle skipping depends on offset and frequency (half-period rule) [VirieuxOperto09, Guasch20]. |
| (none) | **Add**: (data residual, PDE residual) trajectory plot per band | Shows whether data fit is being bought with PDE violation [Aghamiry19 Fig. 3]. |
| (none) | **Add (if wavefield truth is cheap)**: relative wavefield error ‖u−u*‖/‖u*‖ | Reported alongside model error by WRI [Aghamiry19]. Depends on #51's save rule. |
| `rel_c_error` | **Change**: keep, but also report on the brain (intracranial) mask; or normalise by ‖c* − c_background‖ | Whole-interior ‖c*‖ is dominated by background. The starting model already scores a few % [Lucka22, 2.18% initial]. |
| (none) | **Add**: RMSE in m/s per region (brain, skull, lesion if any) | The transcranial standard [Robins23]. Physically interpretable. |
| (none) | **Add**: region mean ± std (bias) per tissue, esp. skull | Catches "skull not updated from start" [Robins23, Mitcham25]. |
| (none) | **Add (only if lesions in phantoms)**: lesion contrast / CNR, or a task score | SSIM and RMSE can disagree with task performance [Lozenski24, Mitcham25]. |
| `ssim_head_roi` | **Change**: record all SSIM settings in the run. Consider Wang-2004 settings (`gaussian_weights=True, sigma=1.5, use_sample_covariance=False`). Decide whether windows straddling the skull boundary count. | SSIM is parameter-dependent and must be reported with its settings [skimage, Breger24]. Edges and low-variance regions are weak spots [PambrunNoumeir15]. |
| `ssim_reason` | **Keep** | Bookkeeping. |
| `c_min`, `c_max`, `c_mean`, `c_std` | **Keep** (cheap sanity/physicality checks) | Not a quality metric; flags blow-ups. |
| `n_evaluations`, `wall_s` | **Keep** | Needed for equal-cost sweep comparisons (error vs time / evaluations) [Lucka22, Karnakov24]. |
| (run-level) | **Add**: inverse-crime flag (same grid/operator for data and inversion?) and starting-model / skull-prior description | Metrics are only meaningful with these conditions stated [Wirgin04, KaipioSomersalo07, Robins23]. |
| (sweep-level) | **Add**: per-band boundary markers on curves; seed scatter once noise exists | Error can rise at band transitions [Lucka22]. Seed scatter [Karnakov24]. Repeats are out of scope (#51 "Not yet specified"). |

## Unverified / gaps

- **Cudeiro-Blanco et al. 2022 (UMB), low-frequency ring device.** SSIM + per-region mean ± std comes from a search-result snippet only; ScienceDirect and PubMed blocked full text.
- **Shah, Warner et al. 2012 (SEG), *Quality assured full-waveform inversion*.** Exists (Imperial listing); its method (phase-residual maps) is not verified.
- **Marty, Boehm, Fichtner (ETH) transcranial FWI papers (SPIE 2021, 2023).** Only abstracts seen; their metrics are not verified.
- **Rasht-Behesht et al. 2022 PINN-FWI (JGR).** Only the abstract seen; their metrics are not verified.
- **Lozenski et al.** RMSE/SSIM masks and SSIM settings are not stated in the paper.
- **Robins et al.** The RMS mask is implied ("brain model") but not defined.
- **Guasch et al. 2020.** The noise-robustness claims in its Supplementary Information were not read.
- **No cited USCT/transcranial paper reports** a relative (‖d_obs‖-normalised) data misfit, a residual-by-offset curve, or per-band misfit numbers. Those suggestions are our inference from general FWI theory.
- **No primary source found that uses** a joint ODIL formulation for ultrasound/transcranial FWI. WRI/IR-WRI is the closest published analogue.

## Sources

- [Guasch20] Guasch L., Calderón Agudo O., Tang M.-X., Nachev P., Warner M. "Full-waveform inversion imaging of the human brain." *npj Digit. Med.* 3, 28 (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7060331/
- [Robins23] Robins T.C., Cueto C., Cudeiro J., Bates O., Calderón Agudo O., Strong G., Guasch L., Warner M., Tang M.-X. "Dual-Probe Transcranial Full-Waveform Inversion: A Brain Phantom Feasibility Study." *Ultrasound Med. Biol.* 49(10):2302–2315 (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC7616382/
- [Mitcham25] Mitcham T., Ali R., Singh M., et al., Duric N. "Transcranial ultrasound tomography for brain imaging: Ex vivo results and potential for stroke imaging." *Med. Phys.* 52(10):e18090 (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12882777/
- [CudeiroBlanco22] Cudeiro-Blanco J. et al. "Design and Construction of a Low-Frequency Ultrasound Acquisition Device for 2-D Brain Imaging Using Full-Waveform Inversion." *Ultrasound Med. Biol.* (2022). https://www.sciencedirect.com/science/article/pii/S0301562922004100 (not read; see Unverified)
- [Lucka22] Lucka F., Pérez-Liva M., Treeby B.E., Cox B.T. "High resolution 3D ultrasonic breast imaging by time-domain full waveform inversion." *Inverse Problems* 38(2):025008 (2022). https://arxiv.org/abs/2102.00755
- [Lozenski24] Lozenski L., Wang H., Li F., Anastasio M., Wohlberg B., Lin Y., Villa U. "Learned Full Waveform Inversion Incorporating Task Information for Ultrasound Computed Tomography." *IEEE Trans. Comput. Imaging* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10491310/ (arXiv:2308.16290)
- [UPFWI22] Jin P., Zhang X., Chen Y., Huang S.X., Liu Z., Lin Y. "Unsupervised Learning of Full-Waveform Inversion: Connecting CNN and Partial Differential Equation in a Loop." ICLR 2022. https://arxiv.org/abs/2110.07584
- [VirieuxOperto09] Virieux J., Operto S. "An overview of full-waveform inversion in exploration geophysics." *Geophysics* 74(6):WCC127–WCC152 (2009). https://doi.org/10.1190/1.3238367 (open copy: https://hal.archives-ouvertes.fr/hal-00457989v1)
- [Aghamiry19] Aghamiry H.S., Gholami A., Operto S. "Improving full-waveform inversion by wavefield reconstruction with the alternating direction method of multipliers." *Geophysics* (2019); arXiv:1809.00891. https://arxiv.org/abs/1809.00891
- [Karnakov24] Karnakov P., Litvinov S., Koumoutsakos P. "Solving inverse problems in physics by optimizing a discrete loss: Fast and accurate learning without neural networks." *PNAS Nexus* 3(1):pgae005 (2024). https://arxiv.org/abs/2205.04611
- [WangTengPerdikaris20] Wang S., Teng Y., Perdikaris P. "Understanding and mitigating gradient pathologies in physics-informed neural networks." arXiv:2001.04536. https://arxiv.org/abs/2001.04536
- [Krishnapriyan21] Krishnapriyan A.S., Gholami A., Zhe S., Kirby R.M., Mahoney M.W. "Characterizing possible failure modes in physics-informed neural networks." NeurIPS 2021. https://arxiv.org/abs/2109.01050
- [Wang04] Wang Z., Bovik A.C., Sheikh H.R., Simoncelli E.P. "Image quality assessment: from error visibility to structural similarity." *IEEE Trans. Image Process.* 13(4):600–612 (2004). https://doi.org/10.1109/TIP.2003.819861
- [WangBovik09] Wang Z., Bovik A.C. "Mean squared error: Love it or leave it?" *IEEE Signal Process. Mag.* 26(1):98–117 (2009). https://doi.org/10.1109/MSP.2008.930649
- [PambrunNoumeir15] Pambrun J.F., Noumeir R. "Limitations of the SSIM quality metric in the context of diagnostic imaging." IEEE ICIP 2015. https://doi.org/10.1109/ICIP.2015.7351345
- [Breger24] Breger A., Biguri A., et al., Schönlieb C.-B. "A study of why we need to reassess full reference image quality assessment with medical images." arXiv:2405.19097. https://arxiv.org/abs/2405.19097
- [skimage] scikit-image `skimage.metrics` API (structural_similarity, normalized_root_mse, peak_signal_noise_ratio). https://scikit-image.org/docs/stable/api/skimage.metrics.html
- [FichtnerTrampert11] Fichtner A., Trampert J. "Resolution analysis in full waveform inversion." *Geophys. J. Int.* 187(3):1604–1624 (2011). https://doi.org/10.1111/j.1365-246X.2011.05218.x
- [FichtnerVanLeeuwen15] Fichtner A., van Leeuwen T. "Resolution analysis by random probing." *J. Geophys. Res. Solid Earth* (2015). https://doi.org/10.1002/2015JB012106
- [Wirgin04] Wirgin A. "The inverse crime." arXiv:math-ph/0401050 (2004). https://arxiv.org/abs/math-ph/0401050
- [KaipioSomersalo07] Kaipio J., Somersalo E. "Statistical inverse problems: Discretization, model reduction and inverse crimes." *J. Comput. Appl. Math.* 198(2):493–504 (2007). https://doi.org/10.1016/j.cam.2005.09.027
