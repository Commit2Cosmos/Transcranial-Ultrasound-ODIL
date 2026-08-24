import pytest
import torch
from matplotlib.colors import TwoSlopeNorm

from odil_wave import VelocityModel
from odil_wave.models import velocity_norm


# --------------------------------------------------------------------------- #
# Analytic profiles
# --------------------------------------------------------------------------- #
def test_homogeneous(grid):
    """A homogeneous medium is a constant field equal to ``base``."""
    vm = VelocityModel(grid, profile="homogeneous", base=1500.0)
    assert vm.c.shape == grid.shape
    assert torch.all(vm.c == 1500.0)
    assert vm.c_min == 1500.0 and vm.c_max == 1500.0
    # no skull masks on a homogeneous profile.
    assert vm.head_mask is None


def test_overdensity_anomaly(grid):
    """Overdensity adds a circular contrast blob to the background."""
    vm = VelocityModel(
        grid,
        profile="overdensity",
        base=1500.0,
        contrast=200.0,
        radius=0.012,
        center=(0.025, 0.025),
    )
    assert vm.c_max == pytest.approx(1700.0)
    assert vm.c_min == pytest.approx(1500.0)
    # the anomaly occupies some, but not all, of the grid.
    n_hot = int((vm.c > 1500.0).sum())
    assert 0 < n_hot < grid.nx * grid.ny


def test_shepp_logan_masks_and_skull(grid):
    """Shepp-Logan builds head/interior/rim masks with a ~3000 m/s skull rim."""
    vm = VelocityModel(grid, profile="shepp_logan")
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
    vm = VelocityModel(grid, profile="shepp_logan_skull", skull_alpha=0.0)
    assert vm.c_min == pytest.approx(1500.0)
    assert vm.c_max == pytest.approx(1500.0)


def test_shepp_logan_skull_validates_params(grid):
    """skull_alpha in [0,1] and skull_sigma >= 0 are enforced."""
    with pytest.raises(ValueError):
        VelocityModel(grid, profile="shepp_logan_skull", skull_alpha=1.5)
    with pytest.raises(ValueError):
        VelocityModel(grid, profile="shepp_logan_skull", skull_sigma=-1.0)


# --------------------------------------------------------------------------- #
# PML embedding
# --------------------------------------------------------------------------- #
def test_build_full_c_edge_replicates(grid):
    """pml_fill='edge' replicates the interior boundary outward."""
    vm = VelocityModel(grid, profile="homogeneous", base=1500.0, pml_fill="edge")
    ci = (
        torch.arange(grid.interior_nx * grid.interior_ny, dtype=grid.dtype).reshape(
            grid.interior_shape
        )
        + 1500.0
    )
    full = vm.build_full_c(ci)
    assert full.shape == grid.shape
    assert torch.allclose(full[grid.interior_slice], ci)
    # the outermost corner replicates the nearest interior corner value.
    assert float(full[0, 0]) == pytest.approx(float(ci[0, 0]))


def test_build_full_c_constant_fill(grid):
    """pml_fill='constant' pads the ring with the fixed pml_c."""
    vm = VelocityModel(
        grid, profile="homogeneous", base=1500.0, pml_fill="constant", pml_c=1234.0
    )
    ci = torch.full(grid.interior_shape, 1600.0, dtype=grid.dtype)
    full = vm.build_full_c(ci)
    assert float(full[0, 0]) == pytest.approx(1234.0)
    assert torch.allclose(full[grid.interior_slice], ci)


def test_pml_fill_replicate_alias(grid):
    """'replicate' is accepted as an alias for 'edge'."""
    vm = VelocityModel(grid, profile="homogeneous", pml_fill="replicate")
    assert vm.pml_fill == "edge"


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
        VelocityModel(grid, profile="does_not_exist")


def test_skull_placeholder_not_implemented(grid):
    """The placeholder 'skull' profile raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        VelocityModel(grid, profile="skull")


def test_bad_pml_fill_raises(grid):
    """An unknown PML-fill mode is rejected."""
    with pytest.raises(ValueError):
        VelocityModel(grid, profile="homogeneous", pml_fill="bogus")


def test_masks_require_skull_profile(grid):
    """skull_region_masks() is unavailable on maskless profiles."""
    vm = VelocityModel(grid, profile="homogeneous")
    with pytest.raises(ValueError):
        vm.skull_region_masks()


def test_velocity_norm_two_slope():
    """velocity_norm returns a TwoSlopeNorm split at vcenter."""
    norm = velocity_norm(vmin=1400.0, vcenter=1600.0, vmax=3000.0)
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vcenter == 1600.0
