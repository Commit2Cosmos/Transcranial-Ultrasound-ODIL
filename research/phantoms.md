# Which numerical phantom should be the true model?

Question (#137, child of map #129): which numerical phantom should the runs use as the true model? The options are:

1. Today's resized-image Shepp-Logan with a rim eroded in cells.
2. An analytic Shepp-Logan rasterised at physical coordinates.
3. The transcranial benchmark of Aubry et al. 2022.
4. A 2D slice of a CT-derived skull.

The ticket's criteria are resolution independence (#84, #80 link 2), brain and skull region masks for the tape's metrics, skull thickness against wavelength (#124 row 12), realism against a clean inverse-crime test bed, availability and licensing, and whether `truth.skull_sigma` and `metrics.ssim.sigma` should move to metres. #84 is decided from this file.

Scope: `VelocityModel` (`odil_wave/models/velocity_models.py`), `ModelCfg`, `SSIMCfg` and `validate_config` (`odil_wave/experiment/config.py`), and the truth and observation grids in `odil_wave/experiment/problem.py`. Material values are cross-checked against [velocity-box-bounds.md](velocity-box-bounds.md) (**VBB**) §7.

Domain terms follow `GLOSSARY.md`: **True model**, **Inverse crime**, **Head region**, **Skull region**, **Brain region**, **Interior**, **PML**, **Baseline**, **Band**.

Every claim carries one provenance tag:

- **[S]**: established by a source. The key points to Sources. "(abstract)" means I reached only the abstract or a search summary. "(via X)" means I read the claim only as X reports it.
- **[R]**: read from this repo's code, an issue or an experiment record.
- **[P]**: measured in this session by a throwaway script on the repo's code at `01d55fe` (§2.2). Not committed.
- **[I]**: my own inference or derivation.

## 1. Conclusion

**Recommendation: option 2, an analytic Shepp-Logan defined in metres.** Change the geometry only and keep today's velocity values, so the 125² Baseline problem barely moves. Move `truth.skull_sigma` and `metrics.ssim.sigma` to metres in the same change. Keep options 3 and 4 as later realism steps, not as the true model now.

Six results decide it.

1. **Today's phantom is not one physical problem across resolutions, and its brain region is contaminated.** The rim is `max(2, round(0.02·phantom px))` cells, so it is 4.0 mm at 125²–375² but 5.3, 7.9 and 16 mm at 96², 64² and 32² [P] (#84 has the 32² row at scale 0.45). The image phantom carries its own skull ellipse, which the Shepp-Logan table makes 0.0644 units thick at the top against 0.0276 elsewhere [S: KakSlaney Table 3.1]. At scale 0.85 over 0.25 m that is 6.8 mm against 2.9 mm [I]. The 4 mm erosion doesn't cover the thick top, so 2–3% of brain-region cells at 125²–375² hold a 1750 m/s skull fragment [P]. `rmse_brain` and `ssim_brain` score these as brain tissue.
2. **An analytic phantom is the standard way to get resolution independence.** "A method to obtain resolution-independent simulations is to formulate the simulation analytically in the continuous domain. This approach goes back to Shepp and Logan" [S: GuerquinKern12 §I]. The phantom is ten ellipses with closed-form parameters [S: KakSlaney Table 3.1]. An ellipse sized in metres stays the same object at every `h`. Only its discretisation changes [I].
3. **The analytic form keeps the clean inverse-crime test bed and adds a way out of it.** Rasterised simulations are "tainted by an aliasing component", and the inverse crime, where "exactly the same discrete model is used for simulation and reconstruction, leads to artificially good results" [S: GuerquinKern12 abstract, §I]. A continuous truth can be sampled on the inversion grid, which keeps the inverse crime #80 assumes, or on a finer observation grid. Today the observation grid can differ only in PML width [R: `problem.py`]. A resized image can do neither cleanly [I].
4. **The benchmark is a forward-model intercomparison, not an imaging phantom.** Aubry22 benchmarks 1–6 are flat or curved plates in a 120 × 70 mm domain driven at 500 kHz by one focused bowl or piston. Benchmarks 7–9 are 3D skull meshes from the MNI152 MRI template (via SimNIBS) as homogeneous cortical bone [S: Aubry22 §II]. None is a closed 2D head with brain features for a ring array [I]. Its **Table 1 tissue values** and its **parietal layering** (1.5 mm cortical, 4 mm trabecular, 1 mm cortical, 6.5 mm in all) are the reusable parts [S: Aubry22 Table I, §II F]. Its data is CC BY 4.0 but ships as one 25.3 GB zip [S: AubryZenodo].
5. **A CT- or atlas-derived slice is realistic but costly, and the obvious sources can't live in the repo.** Guasch et al. built their true model from MIDA at 0.5 mm [S: Guasch20 Methods]. MIDA's licence forbids distributing "the Model Data or works derived from the Model Data … in the original form or in any modified or updated form" [S: MIDAlicence §2.3.2]. So a slice can't be committed. CQ500 head CT is CC BY-NC-SA [S: Qure (via blog)]. The CT-to-speed map moves with X-ray energy and reconstruction method, and CT explains only about half the variance of skull speed [S: Webb18 (abstract)]. A CT slice also needs segmentation for masks, and roughly `h ≤ 1 mm` (≥ 250² nodes) to carry its detail [I].
6. **SSIM's scale and the skull blur are physical lengths, so cells confound cross-resolution comparisons.** "The right scale depends on viewing conditions … A single-scale method … may be appropriate only for specific settings" [S: WangMS03 §1, §3.1]. At 125² `ssim.sigma = 1.5` cells is 3.0 mm with a 22 mm window. At 250² it is 1.5 mm with an 11 mm window [I, from `ssim_window_size`]. So #80 link 2 would compare SSIM at two different scales. `skull_sigma` (1–3 cells, 2–6 mm at 125²) changes the truth itself with `h` [R: `ModelCfg` docstring] [I].

### Recommendations

| # | What | Why | Evidence | Cost |
|---|---|---|---|---|
| 1 | **Analytic Shepp-Logan in metres.** Draw the ten ellipses of KakSlaney Table 3.1 (rotated so the long axis lies along `y`, as today), scaled by `scale · interior extent`. Assign each region a speed directly, not through image intensity. | Results 1–3. The phantom becomes one continuous object, and every grid samples it. | S: GuerquinKern12, KakSlaney; P | M |
| 2 | **Skull as a length.** Add `truth.skull_thickness` in metres. The brain ellipse is the head ellipse with semi-axes reduced by `T`, centred, which drops Table 3.1's −0.0184 offset. Default `T = 4.0 mm`, the current rim on the axes at 125² [R: #84]. Warn when `T < 2h`, like `sigma_s`'s warning. | Answers #84's "rim in metres". The default keeps the Baseline's skull. Dropping the offset removes the thick-top fragment (result 1). | R; I | S |
| 3 | **Cell-averaged rasterisation.** Set each cell's `m = 1/c²` to the mean over the cell of the continuous phantom (supersampled, e.g. 8 × 8), not to the value at the node. Masks: skull = cells whose skull fraction is ≥ 0.5, brain = cells fully inside the brain ellipse. | Cross-resolution truths are then projections of one function. Averaging `1/c²` is the long-wavelength effective medium of a constant-density wave equation [I]. Below `T ≈ 2h` the skull becomes a partial-volume ring, not a different-thickness ring. | I | S |
| 4 | **Keep today's velocity map** for the change itself: water 1500, ventricles 1450, brain 1510, small features 1540/1570, skull 3000 m/s [P, from today's intensity map]. | Keeps the change to geometry only, so `current` experiment records stay comparable. Aubry22's values (brain 1560, cortical 2800, trabecular 2300 m/s) are a separate Baseline change (new question A). | R; S: Aubry22 Table I | none |
| 5 | **`truth.skull_sigma` and `metrics.ssim.sigma` in metres**, in the same change. Defaults that reproduce 125²: `ssim.sigma = 3.0e-3` m (1.5 × 2.016 mm) and `skull_sigma = 0`. `validate_config` checks the window in cells (≥ 7, fits the interior) after conversion. | Result 6. #80 link 2 needs both fixed in metres. | S: WangMS03; R | S |
| 6 | **Options 3 and 4 as later realism stages.** First, layered skull values from Aubry22 on the analytic geometry. Then a 2D slice of the Aubry22 BM8 skull mask, the only redistributable skull geometry found (CC BY 4.0). Not MIDA, not CQ500. | Results 4–5. | S | M–L |

**Not recommended now** [I]:

- **Option 1 with only a documented floor** (#84's "floor documented only"). It leaves the brain-region fragment (result 1) and keeps tests on small grids confounded.
- **Option 1 with only the rim in metres.** The rest of the phantom is still an image resampled per grid, so the brain features alias differently at each `h`, and the image's own skull ellipse still overlaps the eroded rim.
- **A CT slice as the default true model.** Licence (result 5), a grid at least 4× larger than the 125² Baseline, and a CT-to-speed map that is itself uncertain [S: Webb18 (abstract)]. That is three new variables at once, against #129's aim of moving fidelity one link at a time.

## 2. Where the code stands

### 2.1 Today's phantom

- **Construction.** `skimage.data.shepp_logan_phantom()` (400 × 400; its docstring cites only Shepp & Logan 1974) is rotated 90°, resized to `round(scale·interior_shape)` px with anti-aliasing, normalised to [0, 1], and padded into the interior [R: `_shepp_logan_interior_phantom`].
- **Intensities.** The image's distinct values are {0, 0.1, 0.2, 0.3, 0.4, 1} [P]. These are the contrast-enhanced "modified" intensities, not Table 3.1's 2.0 / −0.98 / −0.02 / 0.01, where the brain features differ by 1% [S: KakSlaney Table 3.1] [I]. MATLAB calls its modified variant one "in which the contrast is improved for better visual perception" [S: MATLABphantom].
- **Masks.** head = `fill_holes(phantom > 0.05)`. brain = head eroded by `max(2, round(0.02·min(phantom px)))` iterations of the default (cross-shaped) structure. skull = head minus brain [R]. The cross structure makes the rim anisotropic: 4.0 mm on the axes, 2.8–5.7 mm oblique [R: #124].
- **Speeds.** `shepp_logan`: `1450 + 300·phantom` inside the head, skull rim 3000, water 1500 [R]. `shepp_logan_skull`: rim only, optionally Gaussian-blurred by `skull_sigma` **cells** and scaled by `skull_alpha` [R].
- **Parity.** The module mirrors Stride's `stride_forward_shepp.py::shepp_logan_sos` "so the Shepp-Logan model built here is numerically identical to the Stride forward ground truth" [R: module comment]. Any new phantom breaks that parity unless the Stride side gets the same function [I].
- **Observation grid.** The truth for the observed data is built on a forward grid that may differ from the inversion grid only in PML width, with `nt` and `dt` shared [R: `problem.py`]. So every run is an inverse crime in the interior [R: record 0002 "Assumptions"].
- **Metric scales.** `SSIMCfg.sigma = 1.5` cells; window `2·int(3.5σ + 0.5) + 1` [R]. `ModelCfg.skull_sigma` "≥ 0 grid cells … (try 1–3)" [R].

### 2.2 Pilot: today's truth across resolutions

Interior 0.25 m square, `scale = 0.85`, default `shepp_logan` profile [P]:

| interior | `h` [mm] | phantom px | erosion | rim on axes [mm] | brain-region speeds [m/s] | brain cells > 1650 m/s |
|---|---|---|---|---|---|---|
| 32² | 8.06 | 27 | 2 | 16.1 | 1450–1647 | 0 |
| 64² | 3.97 | 54 | 2 | 7.9 | 1450–1737 | 0.9% |
| 96² | 2.63 | 82 | 2 | 5.3 | n/a | n/a |
| 125² | 2.02 | 106 | 2 | 4.0 | 1450–1750 | 3.0% |
| 250² | 1.00 | 212 | 4 | 4.0 | 1450–1750 | 2.0% |
| 375² | 0.67 | 319 | 6 | 4.0 | 1450–1750 | 1.9% |

The 1750 m/s cells are `1450 + 300·1.0`, the image's skull intensity, inside the brain mask [P] [I]. The rim area is 19.4–19.7 cm² from 125² up, and 39 and 81 cm² at 64² and 32² [P].

## 3. The options against the criteria

| Criterion | 1. Image SL, eroded rim | 2. Analytic SL in metres | 3. Aubry22 benchmark | 4. CT-derived slice |
|---|---|---|---|---|
| Resolution independence | No below ~100 px phantom; brain features alias per grid [R: #84] [P] | Yes by construction [S: GuerquinKern12] | Geometry is continuous (STL) [S: Aubry22 §II F] | Fixed at the scan voxel (0.5 mm MIDA [S: Guasch20]); resampled per grid |
| Brain and skull masks | Yes, but the brain mask holds skull cells (§2.2) [P] | Exact, from the ellipses [I] | Skull only. BM7–9 are homogeneous bone in water or brain [S: Aubry22] | Needs segmentation; MIDA ships it, CT doesn't [S: MIDAlicence §1] [I] |
| Skull thickness vs wavelength | 4 mm (2.9–6.8 mm in the image's own ellipse) | A parameter `T` in metres | 6.5 mm parietal [S: Aubry22 §II F] | Anatomical, about 7 mm [S: Guasch20] |
| Realism | Low: single-speed rim, toy brain | Low, same as option 1 | Medium: layered bone, real values | High |
| Clean inverse-crime test bed | Yes, on its own grid only | Yes, plus an off-grid truth option | n/a as a ring-array phantom | Possible, but large grid |
| Availability, licence | skimage (BSD-3) [R] | Ten published ellipses [S: KakSlaney] | CC BY 4.0, 25.3 GB zip [S: AubryZenodo] | MIDA: no redistribution [S: MIDAlicence]; CQ500: CC BY-NC-SA [S: Qure] |
| Cost to adopt | none | S–M | M (2D slice + new values) | L |

**Skull thickness against wavelength.** At 80 kHz the wavelength is 37.5 mm in a 3000 m/s skull and 35 mm in 2800 m/s cortical bone [I]. A 4 mm skull is `λ/9.4`, matching #124 row 12 [R]. A 6.5 mm parietal skull is `λ/5.8` at 3000 m/s [I]. #124's λ/8 detection rule of thumb is unverified [R: #124]. Under that rule the 6.5 mm skull is detectable at 80 kHz (it needs `f ≥ 58 kHz`) and the 4 mm skull isn't (it needs `f ≥ 94 kHz`) [I]. So the thickness default is a physics choice, not a numerics one. That is why recommendation 2 makes it a parameter and keeps 4 mm only for Baseline continuity.

**Resolving the skull.** A cell-averaged skull of thickness `T` is a resolved layer only when `T ≳ 2h` [I]. For 4 mm that is `h ≤ 2 mm`, about 126 nodes over 0.25 m: the default 125² sits at `T = 1.98 h`. Below that, small test grids see a partial-volume ring of reduced peak speed. That is the same continuous problem, under-resolved, and not a different problem, which is what #84 asks for [I].

**Is a 2D slice physically sound?** In Guasch20, "2D FWI of 2D data" recovers a model "as accurate as that recovered by 3D inversion of 3D data", while 2D FWI of 3D data fails [S: Guasch20 Results]. All four options here are 2D truths with 2D data, so out-of-plane energy is not an issue for any of them [I].

## 4. Implications for #84

Decide #84 as **"resolution-independent phantom", with the rim in metres and both sigmas in metres in the same change** (recommendations 1, 2, 3 and 5). The tape's metrics keep a Shepp-Logan truth with brain and skull regions, and #80 link 2 (125² vs 250²) then compares one physical problem at one SSIM scale.

The change will move the true model slightly at 125², because the brain-region skull fragment and the anisotropic rim go away. `current` records on a Shepp-Logan truth (record 0002) need the one-line note from "Maintaining the bank" [R: CLAUDE.md].

## 5. Open questions and unverified

**New questions that could become tickets** [I]:

- **A. Tissue values.** Should the Baseline move to Aubry22's values (brain 1560, cortical 2800, trabecular 2300 m/s) and a layered skull? It changes `c_max` headroom (VBB §7 recommendation 5) and the soft-tissue contrast.
- **B. Skull thickness default.** 4 mm (continuity) or 6.5 mm (parietal, Aubry22)? It decides whether #124 row 12 passes at 80 kHz.
- **C. Off-grid observed data.** Generate observed data from the analytic truth on a finer forward grid, to leave the inverse crime. Today the forward grid differs only in PML width.
- **D. Stride parity.** Does anything still depend on numerical parity with Stride's `shepp_logan_sos`? If so, port the analytic phantom there.

**Unverified:**

- Shepp & Logan 1974 itself was not read. The ellipse table is from Kak & Slaney Table 3.1, which cites it. The contrast-enhanced intensities are inferred from skimage's image values, not from Toft or skimage documentation.
- Webb18 was read at abstract level only, including the "about half the variance" figure.
- The CQ500 licence is from qure.ai's blog, not the dataset page.
- The Aubry22 Zenodo archive's contents (whether 2D-sliceable skull masks are included at what resolutions) were not downloaded. The paper says grid masks at "a range of resolutions" and the STL files are "available alongside the simulations results" [S: Aubry22 §II F].
- The λ/8 thin-layer threshold is #124's rule of thumb, not checked here.
- Recommendation 3's `1/c²` averaging rule is my derivation for constant density; no source checked for this repo's PDE.

**Read in full or in the cited sections:** Aubry22 §II and Table I, GuerquinKern12 abstract, §I and §V, KakSlaney Ch. 3 pp. 50–55 and Table 3.2, WangMS03 §1–3.1, Guasch20 (full text via Europe PMC), MIDAlicence (full).

## Sources

- [KakSlaney] Kak A.C., Slaney M. *Principles of Computerized Tomographic Imaging*, IEEE Press 1988 / SIAM 2001, Ch. 3, Table 3.1 ("Summary of parameters for tomography simulations"). Read. https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf
- [GuerquinKern12] Guerquin-Kern M., Lejeune L., Pruessmann K.P., Unser M. "Realistic analytical phantoms for parallel magnetic resonance imaging." *IEEE Trans. Med. Imaging* 31(3):626–636 (2012). Read abstract, §I, §V. https://bigwww.epfl.ch/publications/guerquinkern1201.pdf
- [Aubry22] Aubry J.-F. et al. "Benchmark problems for transcranial ultrasound simulation: intercomparison of compressional wave models." *J. Acoust. Soc. Am.* 152(2):1003–1019 (2022). Read §II and Table I. https://arxiv.org/abs/2202.04552
- [AubryZenodo] Aubry J.-F. et al. Intercomparison dataset v1.0, Zenodo (2022), CC BY 4.0, `INTERCOMPARISON-PH1-DATA-V1.0.zip` 25.3 GB. Read the record page. https://doi.org/10.5281/zenodo.6020543
- [Guasch20] Guasch L., Calderón Agudo O., Tang M.-X., Nachev P., Warner M. "Full-waveform inversion imaging of the human brain." *npj Digit. Med.* 3:28 (2020). Read the full text via Europe PMC (PMC7060331). https://doi.org/10.1038/s41746-020-0240-8
- [MIDAlicence] IT'IS Foundation, MIDA Model user licence agreement. Read. https://itis.swiss/assets/Downloads/VirtualPopulation/License_Agreements/LicenseAgreementMIDA.pdf
- [Qure] qure.ai, "Challenges of development & validation of deep learning for radiology" (blog; CQ500 released under CC BY-NC-SA). Search summary. https://qure.ai/blog/challenges-of-development-validation-of-deep-learning-for-radiology
- [Webb18] Webb T.D. et al. "Measurements of the relationship between CT Hounsfield units and acoustic velocity and how it changes with photon energy and reconstruction method." *IEEE TUFFC* 65(7):1111–1124 (2018). Abstract (search summary). PubMed 29993366.
- [WangMS03] Wang Z., Simoncelli E.P., Bovik A.C. "Multi-scale structural similarity for image quality assessment." *Proc. 37th Asilomar Conf.* (2003). Read §1–3.1. https://www.cns.nyu.edu/pub/eero/wang03b.pdf
- [MATLABphantom] MathWorks, `phantom` documentation (Image Processing Toolbox). Read. https://www.mathworks.com/help/images/ref/phantom.html
- [skimage] scikit-image 0.26.0, `skimage.data.shepp_logan_phantom` docstring. Read in the repo's environment.
- [VBB] [velocity-box-bounds.md](velocity-box-bounds.md).
