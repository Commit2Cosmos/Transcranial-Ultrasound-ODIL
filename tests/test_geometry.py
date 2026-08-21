import pytest
import torch

from odil_wave import AcquisitionGeometry
from odil_wave.geometry import source_ring_indices


# --------------------------------------------------------------------------- #
# source_ring_indices
# --------------------------------------------------------------------------- #
def test_source_ring_indices_even_spacing():
    """Sources are an evenly spaced subset of the receiver ring."""
    assert source_ring_indices(8, 4) == [0, 2, 4, 6]
    assert source_ring_indices(8, 3) == [0, 2, 4]
    assert source_ring_indices(16, 4) == [0, 4, 8, 12]


def test_source_ring_indices_step_floored():
    """Requesting more sources than receivers floors the step at 1."""
    assert source_ring_indices(4, 10) == [0, 1, 2, 3]


def test_source_ring_indices_guards_count():
    """A non-positive source count is rejected."""
    with pytest.raises(ValueError):
        source_ring_indices(8, 0)
    with pytest.raises(ValueError):
        source_ring_indices(8, -2)


# --------------------------------------------------------------------------- #
# AcquisitionGeometry
# --------------------------------------------------------------------------- #
def test_ring_on_grid_and_counts(grid, source, freq_selection):
    """Receivers/sources land inside the grid; realised source count is exposed."""
    geom = AcquisitionGeometry(grid, source, freq_selection, n_receivers=8, n_sources=4)
    assert geom.recv_ij.shape == (8, 2)
    assert geom.n_sources == 4
    assert geom.src_ij.shape == (4, 2)
    # every index is a valid full-grid node.
    assert torch.all(geom.recv_ij[:, 0] >= 0) and torch.all(
        geom.recv_ij[:, 0] < grid.nx
    )
    assert torch.all(geom.recv_ij[:, 1] >= 0) and torch.all(
        geom.recv_ij[:, 1] < grid.ny
    )
    # sources are exactly the subsampled receiver nodes.
    idx = geom.source_ring_indices
    assert torch.equal(geom.src_ij, geom.recv_ij[idx])


def test_src_position_matches_grid_axes(geometry, grid):
    """src_position returns the physical coordinates of the source node."""
    x, y = geometry.src_position(0)
    i, j = int(geometry.src_ij[0, 0]), int(geometry.src_ij[0, 1])
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
    geom = AcquisitionGeometry(
        grid,
        source,
        freq_selection,
        n_receivers=8,
        n_sources=2,
        source_spatial="gaussian",
    )
    field_t = geom.source_field_time(0)
    spatial_support = (field_t.abs().sum(dim=0) > 1e-12).sum()
    assert int(spatial_support) > 1
    assert geom.sigma_s > 0


def test_source_field_shapes(geometry, grid, freq_selection):
    """Frequency- and time-domain source fields have matching grid shapes."""
    sf = geometry.source_field(0)
    assert sf.shape == (freq_selection.n_frequencies, grid.nx, grid.ny)
    assert sf.is_complex()

    sft = geometry.source_field_time(0)
    assert sft.shape == (grid.nt, grid.nx, grid.ny)
    assert sft.dtype == grid.dtype


def test_extract_observations_samples_receivers(geometry, grid, freq_selection):
    """extract_observations gathers exactly the receiver nodes of a field."""
    nf = freq_selection.n_frequencies
    U = torch.arange(nf * grid.nx * grid.ny, dtype=grid.dtype).reshape(
        nf, grid.nx, grid.ny
    )
    obs = geometry.extract_observations(U)
    assert obs.shape == (nf, geometry.n_receivers)
    i, j = geometry.recv_ij[:, 0], geometry.recv_ij[:, 1]
    assert torch.equal(obs, U[:, i, j])


def test_extract_observations_batched(geometry, grid, freq_selection):
    """Extraction preserves a leading shot/batch dimension."""
    nf = freq_selection.n_frequencies
    U = torch.randn(3, nf, grid.nx, grid.ny)
    obs = geometry.extract_observations(U)
    assert obs.shape == (3, nf, geometry.n_receivers)
