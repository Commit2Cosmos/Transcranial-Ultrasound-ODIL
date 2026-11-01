import pytest
import torch

from odil_wave import FrequencySelection

from conftest import F0, make_geometry, make_grid, make_source


# --------------------------------------------------------------------------- #
# AcquisitionGeometry
# --------------------------------------------------------------------------- #
def test_ring_on_grid_and_counts(grid, source, freq_selection):
    """Transducers land on valid grid nodes; one shot per transducer."""
    geom = make_geometry(grid, source, freq_selection, source_spatial="gaussian")
    assert geom.n_transducers == 8
    assert geom.n_shots == 8
    assert geom.transducer_ij.shape == (8, 2)
    assert torch.all(geom.transducer_ij[:, 0] >= 0) and torch.all(
        geom.transducer_ij[:, 0] < grid.nx
    )
    assert torch.all(geom.transducer_ij[:, 1] >= 0) and torch.all(
        geom.transducer_ij[:, 1] < grid.ny
    )


@pytest.mark.parametrize("n", [0, 1])
def test_ring_needs_two_transducers(n, grid, source, freq_selection):
    """With fewer than two transducers no shot has a recording transducer."""
    with pytest.raises(ValueError, match="n_transducers"):
        make_geometry(grid, source, freq_selection, n_transducers=n)


def test_every_other_transducer_records_each_shot(geometry):
    """Shot s is recorded by the N - 1 transducers other than s."""
    n = geometry.n_transducers
    assert geometry.receivers.shape == (n, n - 1)
    for s in range(n):
        assert sorted(geometry.receivers[s].tolist()) == [r for r in range(n) if r != s]


def test_default_ring_center_is_interior_center(grid, source, freq_selection):
    """ring_center=None centres the ring on the interior region."""
    geom = make_geometry(grid, source, freq_selection, ring_center=None)
    assert geom.ring_center == pytest.approx(grid.interior_center)
    explicit = make_geometry(grid, source, freq_selection, ring_center=(0.02, 0.03))
    assert explicit.ring_center == (0.02, 0.03)


def test_fixture_ring_lies_in_interior(geometry, grid):
    """Every transducer of the shared fixture sits outside the PML."""
    sx, sy = grid.interior_slice
    ij = geometry.transducer_ij
    assert torch.all((ij[:, 0] >= sx.start) & (ij[:, 0] < sx.stop))
    assert torch.all((ij[:, 1] >= sy.start) & (ij[:, 1] < sy.stop))


def test_src_position_matches_grid_axes(geometry, grid):
    """src_position returns the physical coordinates of the transmitting
    transducer's node."""
    x, y = geometry.src_position(0)
    i, j = int(geometry.transducer_ij[0, 0]), int(geometry.transducer_ij[0, 1])
    assert x == pytest.approx(float(grid.x[i]))
    assert y == pytest.approx(float(grid.y[j]))


def test_point_source_profile_is_single_cell(geometry):
    """A point source injects at exactly one grid node."""
    # geometry fixture uses source_spatial="point".
    field_t = geometry.source_field_time(0)  # (nt, nx, ny)
    spatial_support = (field_t.abs().sum(dim=0) > 0).sum()
    assert int(spatial_support) == 1


def test_gaussian_source_profile_is_spread(grid, source, freq_selection):
    """A Gaussian source spreads its injection over several cells."""
    geom = make_geometry(grid, source, freq_selection, source_spatial="gaussian")
    field_t = geom.source_field_time(0)
    spatial_support = (field_t.abs().sum(dim=0) > 1e-12).sum()
    assert int(spatial_support) > 1
    assert geom.sigma_s > 0


@pytest.mark.parametrize("source_spatial", ["gaussian", "point"])
def test_source_integrates_to_its_amplitude_on_any_grid(source_spatial):
    """The spatial profile is a density: sum(q) dx dy is the source spectrum,
    the same at h and h / 2 (sigma_s = 4 mm is 1.5 h on the coarse grid)."""
    for shape in [(20, 20), (39, 39)]:
        grid = make_grid(interior_shape=shape, pml_width=4 * shape[0] // 20)
        freq = FrequencySelection.from_frequencies(grid, [F0])
        src = make_source(grid)
        geom = make_geometry(grid, src, freq, source_spatial=source_spatial)
        integral = geom.source_field(0).sum(dim=(-2, -1)) * grid.dx * grid.dy
        torch.testing.assert_close(
            integral, src.spectrum(freq).to(integral.dtype), rtol=1e-6, atol=0
        )


def test_source_field_shapes(geometry, grid, freq_selection):
    """Frequency- and time-domain source fields have matching grid shapes."""
    sf = geometry.source_field(0)
    assert sf.shape == (freq_selection.n_frequencies, grid.nx, grid.ny)
    assert sf.is_complex()

    sft = geometry.source_field_time(0)
    assert sft.shape == (grid.nt, grid.nx, grid.ny)
    assert sft.dtype == grid.dtype


def test_extract_observations_skips_the_transmitting_transducer(
    geometry, grid, freq_selection
):
    """Each shot's traces are its N - 1 receivers' nodes, never its own node."""
    n, nf = geometry.n_transducers, freq_selection.n_frequencies
    U = torch.arange(n * nf * grid.nx * grid.ny, dtype=grid.dtype).reshape(
        n, nf, grid.nx, grid.ny
    )
    obs = geometry.extract_observations(U)
    assert obs.shape == (n, nf, n - 1)
    for s in range(n):
        ij = geometry.receiver_ij[s]
        assert torch.equal(obs[s], U[s][:, ij[:, 0], ij[:, 1]])
        own = U[s][:, geometry.transducer_ij[s, 0], geometry.transducer_ij[s, 1]]
        assert not torch.isin(own, obs[s]).any()


def test_extract_observations_batched(geometry, grid):
    """Extraction keeps any dims between the shot axis and the grid axes."""
    n = geometry.n_transducers
    U = torch.randn(n, 3, 2, grid.nx, grid.ny)
    assert geometry.extract_observations(U).shape == (n, 3, 2, n - 1)
