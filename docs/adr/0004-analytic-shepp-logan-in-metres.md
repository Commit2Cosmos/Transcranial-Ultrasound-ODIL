# The Shepp-Logan true model is analytic and sized in metres

The Shepp-Logan profiles sample a continuous phantom: the Kak & Slaney Table 3.1 ellipses scaled to `scale ·` the interior extent, a single-layer skull `truth.skull_thickness` thick, and each cell's `1/c²` averaged over 8 × 8 sub-samples. We chose this over the resized `skimage` image with a rim eroded in cells because that rim's thickness in metres changed below a ~100 px phantom, and its brain region held fragments of the image's own skull. So runs at different resolutions solved different physical problems (#84). The analytic phantom is the same object on every grid, at the cost of a skull that is a partial-volume shell wherever it is under 2 cells thick.

Rejected alternatives:

- **The image phantom with a documented floor**: keeps the brain-region skull fragment and leaves tests on small grids solving a different problem.
- **The image phantom with only the rim in metres**: the brain features still alias differently at each `h`.
- **The Aubry et al. 2022 benchmark or a CT slice**: these are realism steps, not a clean inverse-crime test bed (see `research/phantoms.md`).

## Consequences

- The brain ellipse is the head ellipse shrunk by the skull thickness, centred. This drops Table 3.1's −0.0184 offset, so the skull has one thickness all round.
- Region masks come from the ellipses. The skull region is cells at least half skull, and the brain region is cells wholly inside the brain ellipse, so boundary cells count in neither region's metrics.
- `truth.skull_sigma` and `metrics.ssim.sigma` are lengths in metres, so the skull blur and the SSIM scale are the same physical lengths at every resolution.
- The phantom no longer matches Stride's `shepp_logan_sos`.
