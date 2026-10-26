import pytest
import torch
from matplotlib.colors import TwoSlopeNorm

from odil_wave import VelocityModel
from odil_wave.models import velocity_norm

from conftest import make_model

# Phantom size used by the Shepp-Logan tests (fraction of the interior).
SCALE = 0.90


# --------------------------------------------------------------------------- #
# Analytic profiles
# --------------------------------------------------------------------------- #
def test_homogeneous(grid):
    """A homogeneous medium is a constant field equal to ``base``."""
    vm = make_model(grid)
    assert vm.c.shape == grid.shape
    assert torch.all(vm.c == 1500.0)
    assert vm.c_min == 1500.0 and vm.c_max == 1500.0
    # no skull masks on a homogeneous profile.
    assert vm.head_mask is None


def test_overdensity_anomaly(grid):
    """Overdensity adds a circular contrast blob to the background."""
    vm = make_model(
        grid, "overdensity", contrast=200.0, radius=0.012, center=(0.025, 0.025)
    )
    assert vm.c_max == pytest.approx(1700.0)
    assert vm.c_min == pytest.approx(1500.0)
    # the anomaly occupies some, but not all, of the grid.
    n_hot = int((vm.c > 1500.0).sum())
    assert 0 < n_hot < grid.nx * grid.ny


def test_overdensity_default_center_is_interior_center(grid):
    """Without ``center`` the anomaly is centred on the interior region."""
    vm = make_model(grid, "overdensity", contrast=200.0, radius=0.012)
    hot = vm.c > 1500.0
    cx = float(grid.X[hot].mean())
    cy = float(grid.Y[hot].mean())
    assert (cx, cy) == pytest.approx(grid.interior_center, abs=grid.dx)
    # the whole anomaly lies inside the interior.
    assert int(hot.sum()) == int(hot[grid.interior_slice].sum())


def test_shepp_logan_masks_and_skull(grid):
    """Shepp-Logan builds head/interior/rim masks with a ~3000 m/s skull rim."""
    vm = make_model(grid, "shepp_logan", scale=SCALE)
    head, interior, rim = vm.skull_region_masks()
    assert head.shape == grid.shape
    assert int(rim.sum()) > 0
    assert int(interior.sum()) > 0
    # rim is inside the head, disjoint from the eroded interior.
    assert torch.all(head[rim])
    assert not torch.any(interior & rim)
    # the skull rim reaches the constant skull speed.
    assert vm.c_max == pytest.approx(3000.0, rel=1e-3)
    # head_mask is interior-shaped.
    assert vm.head_mask.shape == grid.interior_shape


def test_shepp_logan_skull_alpha_zero_is_water(grid):
    """skull_alpha=0 removes all contrast, leaving the water background."""
    vm = make_model(
        grid, "shepp_logan_skull", scale=SCALE, skull_alpha=0.0, skull_sigma=0.0
    )
    assert vm.c_min == pytest.approx(1500.0)
    assert vm.c_max == pytest.approx(1500.0)


def test_shepp_logan_skull_validates_params(grid):
    """skull_alpha in [0,1] and skull_sigma >= 0 are enforced."""
    with pytest.raises(ValueError):
        make_model(
            grid, "shepp_logan_skull", scale=SCALE, skull_alpha=1.5, skull_sigma=0.0
        )
    with pytest.raises(ValueError):
        make_model(
            grid, "shepp_logan_skull", scale=SCALE, skull_alpha=1.0, skull_sigma=-1.0
        )


# --------------------------------------------------------------------------- #
# PML embedding
# --------------------------------------------------------------------------- #
def _pml_ring(grid):
    """Boolean full-grid mask of the PML ring (everything outside the interior)."""
    ring = torch.ones(grid.shape, dtype=torch.bool)
    ring[grid.interior_slice] = False
    return ring


def test_build_full_c_constant_fill(grid):
    """The whole PML ring is padded with the fixed pml_c, not the interior edge."""
    vm = make_model(grid, pml_c=1234.0)
    # Non-uniform interior so an edge-replicating fill would differ from pml_c.
    ci = (
        torch.arange(grid.interior_nx * grid.interior_ny, dtype=grid.dtype).reshape(
            grid.interior_shape
        )
        + 1600.0
    )
    full = vm.build_full_c(ci)
    assert full.shape == grid.shape
    assert torch.allclose(full[grid.interior_slice], ci)
    assert torch.all(full[_pml_ring(grid)] == 1234.0)


def test_from_field_build_full_c_pads_with_pml_c(grid):
    """from_field models use the same constant pml_c fill."""
    vm = VelocityModel.from_field(
        grid, torch.full(grid.shape, 1600.0, dtype=grid.dtype), pml_c=1500.0
    )
    ci = torch.full(grid.interior_shape, 1700.0, dtype=grid.dtype)
    full = vm.build_full_c(ci)
    assert torch.allclose(full[grid.interior_slice], ci)
    assert torch.all(full[_pml_ring(grid)] == 1500.0)


def test_from_field_custom_profile(grid):
    """from_field wraps an arbitrary field as a maskless 'custom' model."""
    c = torch.full(grid.shape, 1600.0, dtype=grid.dtype)
    vm = VelocityModel.from_field(grid, c)
    assert vm.profile == "custom"
    assert torch.allclose(vm.c, c)
    assert vm.pml_c == pytest.approx(1600.0)  # defaults to the field minimum
    assert vm.head_mask is None


# --------------------------------------------------------------------------- #
# Error handling / helpers
# --------------------------------------------------------------------------- #
def test_unknown_profile_raises(grid):
    """An unrecognised profile is rejected."""
    with pytest.raises(ValueError):
        make_model(grid, "does_not_exist")


def test_masks_require_skull_profile(grid):
    """skull_region_masks() is unavailable on maskless profiles."""
    vm = make_model(grid)
    with pytest.raises(ValueError):
        vm.skull_region_masks()


def test_velocity_norm_two_slope():
    """velocity_norm returns a TwoSlopeNorm split at vcenter."""
    norm = velocity_norm(vmin=1400.0, vcenter=1600.0, vmax=3000.0)
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vcenter == 1600.0
