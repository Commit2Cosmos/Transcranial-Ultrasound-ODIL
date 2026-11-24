import numpy as np
import pytest
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm

from odil_wave import VelocityModel
from odil_wave.models import profile_velocity_norm, velocity_norm

from conftest import make_grid, make_model

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
    assert vm.head_mask is None


def test_overdensity_anomaly(grid):
    """Overdensity adds a circular contrast blob to the background."""
    vm = make_model(
        grid, "overdensity", contrast=200.0, radius=0.012, center=(0.025, 0.025)
    )
    assert vm.c_max == pytest.approx(1700.0)
    assert vm.c_min == pytest.approx(1500.0)
    n_hot = int((vm.c > 1500.0).sum())
    assert 0 < n_hot < grid.nx * grid.ny


def test_overdensity_default_center_is_interior_center(grid):
    """Without ``center`` the anomaly is centred on the interior region."""
    vm = make_model(grid, "overdensity", contrast=200.0, radius=0.012)
    hot = vm.c > 1500.0
    cx = float(grid.X[hot].mean())
    cy = float(grid.Y[hot].mean())
    assert (cx, cy) == pytest.approx(grid.interior_center, abs=grid.dx)
    assert int(hot.sum()) == int(hot[grid.interior_slice].sum())


def test_shepp_logan_masks_and_skull(grid):
    """Shepp-Logan builds head/interior/rim masks with a ~3000 m/s skull rim."""
    vm = make_model(grid, "shepp_logan", scale=SCALE)
    head, interior, rim = vm.skull_region_masks()
    assert head.shape == grid.shape
    assert int(rim.sum()) > 0
    assert int(interior.sum()) > 0
    assert torch.all(head[rim])
    assert not torch.any(interior & rim)
    assert vm.c_max == pytest.approx(3000.0, rel=1e-3)
    assert vm.head_mask.shape == grid.interior_shape


# The default config's head (0.25 m interior, scale 0.85), sampled at three
# resolutions spanning a skull thinner than a cell to one 6.5 cells thick.
HEAD_EXTENT = 0.25
HEAD_SCALE = 0.85


def _head(n, **kw):
    g = make_grid(
        interior_shape=(n, n),
        interior_extent=((0.0, HEAD_EXTENT), (0.0, HEAD_EXTENT)),
        c_max=3100.0,
        pml_width=2,
    )
    return make_model(g, "shepp_logan", scale=HEAD_SCALE, **kw)


def _interior_c(vm):
    return vm.c[vm.grid.interior_slice].numpy()


def _cell_at(vm, x, y):
    """The interior cell holding the point ``(x, y)`` (m)."""
    g = vm.grid
    xs, ys = (a[s].numpy() for a, s in zip((g.x, g.y), g.interior_slice))
    return int(np.argmin(abs(xs - x))), int(np.argmin(abs(ys - y)))


def _rim_thickness(vm):
    """The skull thickness (m) on the head's short axis, from the slowness
    ``1/c²`` integrated along the row through the centre: water, then skull,
    then plain brain (1510 m/s) up to half the head's half-width."""
    half = HEAD_SCALE * HEAD_EXTENT / 2
    c0 = HEAD_EXTENT / 2
    a = 0.69 * half  # Table 3.1's short semi-axis
    i_end, j = _cell_at(vm, c0 - 0.5 * half, c0)
    h = vm.grid.dx
    m = _interior_c(vm)[: i_end + 1, j] ** -2.0
    m_water, m_skull, m_brain = 1500.0**-2, 3000.0**-2, 1510.0**-2
    x_end = float(vm.grid.x[vm.grid.interior_slice[0]][i_end]) + h / 2
    excess = float(np.sum(m - m_water) * h)
    brain_span = x_end - (c0 - a)
    return (excess - (m_brain - m_water) * brain_span) / (m_skull - m_brain)


@pytest.mark.parametrize("n", [32, 125, 250])
def test_the_rim_is_as_thick_in_metres_at_any_resolution(n):
    """The default 6.5 mm skull holds at 32², where it is thinner than a cell."""
    vm = _head(n)
    # 8 x 8 sub-samples place each of the skull's two edges to within h/16.
    assert _rim_thickness(vm) == pytest.approx(6.5e-3, abs=vm.grid.dx / 8)


def test_skull_thickness_sets_the_rim_in_metres():
    assert _rim_thickness(_head(125, skull_thickness=5e-3)) == pytest.approx(
        5e-3, abs=0.2e-3
    )


@pytest.mark.parametrize("n", [32, 125, 250])
def test_brain_region_speeds_are_the_same_in_metres_at_any_resolution(n):
    """Plain brain, a ventricle and a feature keep their speeds at fixed points
    in metres, and the brain region holds no skull."""
    vm = _head(n)
    c = _interior_c(vm)
    half = HEAD_SCALE * HEAD_EXTENT / 2
    c0 = HEAD_EXTENT / 2
    for (u, v), speed in [
        ((-0.5, 0.0), 1510.0),
        ((0.22, 0.0), 1450.0),
        ((-0.22, 0.0), 1450.0),
        ((0.0, -0.35), 1540.0),
    ]:
        cell = _cell_at(vm, c0 + u * half, c0 + v * half)
        assert c[cell] == pytest.approx(speed, abs=1e-6), (u, v)
    _, brain, _ = vm.skull_region_masks()
    in_brain = vm.c[brain].numpy()
    assert in_brain.min() >= 1450.0 - 1e-6
    assert in_brain.max() <= 1570.0 + 1e-6


def test_shepp_logan_skull_alpha_zero_is_water(grid):
    """skull_alpha=0 removes all contrast, leaving the water background."""
    vm = make_model(
        grid, "shepp_logan_skull", scale=SCALE, skull_alpha=0.0, skull_sigma=0.0
    )
    assert vm.c_min == pytest.approx(1500.0)
    assert vm.c_max == pytest.approx(1500.0)


def test_skull_sigma_blurs_the_skull_over_metres():
    """A 4 mm blur of the 6.5 mm skull lowers its peak to ~2380 m/s (the
    continuous erf profile) at either resolution."""
    peaks = [
        make_model(
            _head(n).grid, "shepp_logan_skull", scale=HEAD_SCALE, skull_sigma=4e-3
        ).c_max
        for n in (125, 250)
    ]
    assert peaks == pytest.approx([2376.0, 2376.0], rel=0.03)


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


def _ramp_interior(grid):
    """A non-uniform interior c (m/s), so every edge cell differs."""
    n = grid.interior_nx * grid.interior_ny
    return torch.arange(n, dtype=grid.dtype).reshape(grid.interior_shape) + 1600.0


def test_build_full_c_replicates_the_interior_edge(grid):
    """Each PML cell takes the speed of the nearest interior edge cell."""
    ci = _ramp_interior(grid)
    full = make_model(grid).build_full_c(ci)
    assert full.shape == grid.shape
    assert torch.equal(full[grid.interior_slice], ci)
    sx, sy = grid.interior_slice
    p = grid.pml_width
    for i in range(p):
        assert torch.equal(full[i, sy], ci[0])
        assert torch.equal(full[-1 - i, sy], ci[-1])
        assert torch.equal(full[sx, i], ci[:, 0])
        assert torch.equal(full[sx, -1 - i], ci[:, -1])
    assert torch.all(full[:p, :p] == ci[0, 0])
    assert torch.all(full[-p:, -p:] == ci[-1, -1])


def test_build_full_c_sums_the_pml_gradient_onto_the_edge_cells(grid):
    """The gradient of a sum over the full grid counts every cell a PML cell
    copies: 1 inside, 1 + p on an edge, (1 + p)^2 at a corner."""
    ci = _ramp_interior(grid).requires_grad_(True)
    make_model(grid).build_full_c(ci).sum().backward()
    p = grid.pml_width
    assert float(ci.grad[1:-1, 1:-1].max()) == 1.0
    assert float(ci.grad[0, 1:-1].min()) == 1.0 + p
    assert float(ci.grad[0, 0]) == (1.0 + p) ** 2


def test_from_interior_replicates_an_interior_c_into_the_pml(grid):
    """A saved band's interior c becomes a full-grid model, its PML ring the
    interior edge's speed."""
    ci = torch.full(grid.interior_shape, 1700.0, dtype=torch.float32)
    ci[0] = 1650.0
    vm = VelocityModel.from_interior(grid, ci)
    assert vm.c.shape == grid.shape and vm.c.dtype == grid.dtype
    assert torch.all(vm.c[grid.interior_slice] == ci.to(grid.dtype))
    assert torch.all(vm.c[: grid.pml_width, grid.interior_slice[1]] == 1650.0)
    assert torch.all(vm.c[-grid.pml_width :, grid.interior_slice[1]] == 1700.0)


def test_from_field_custom_profile(grid):
    """from_field wraps an arbitrary field as a maskless 'custom' model."""
    c = torch.full(grid.shape, 1600.0, dtype=grid.dtype)
    vm = VelocityModel.from_field(grid, c)
    assert vm.profile == "custom"
    assert torch.allclose(vm.c, c)
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


def bounds(norm) -> tuple:
    if isinstance(norm, TwoSlopeNorm):
        return norm.vmin, norm.vcenter, norm.vmax
    return norm.vmin, norm.vmax


@pytest.mark.parametrize("profile", ["shepp_logan", "shepp_logan_skull"])
def test_shepp_logan_velocity_is_two_slope_at_1400_1600_3000(profile):
    norm = profile_velocity_norm(profile)
    assert isinstance(norm, TwoSlopeNorm)
    assert bounds(norm) == (1400.0, 1600.0, 3000.0)


@pytest.mark.parametrize("profile", ["homogeneous", "overdensity"])
def test_other_velocity_is_linear_and_autoscaled(profile):
    norm = profile_velocity_norm(profile)
    assert type(norm) is Normalize
    # Unset bounds autoscale to the plotted field.
    assert bounds(norm) == (None, None)


def test_explicit_velocity_bounds_override_the_profile():
    assert bounds(profile_velocity_norm("shepp_logan", vmax=2000.0)) == (
        1400.0,
        1600.0,
        2000.0,
    )
    assert bounds(profile_velocity_norm("homogeneous", vmin=1450.0)) == (1450.0, None)
    # A centre makes any profile two-slope; missing bounds are the defaults.
    norm = profile_velocity_norm("homogeneous", vcenter=1555.0)
    assert bounds(norm) == (1400.0, 1555.0, 3000.0)


def shown_norm(ax):
    return ax.get_images()[0].norm


def test_a_shepp_logan_model_shows_on_the_two_slope_scale(grid):
    ax = make_model(grid, "shepp_logan", scale=SCALE).show()
    assert bounds(shown_norm(ax)) == (1400.0, 1600.0, 3000.0)


def test_a_homogeneous_model_shows_on_a_linear_scale(grid):
    assert type(shown_norm(make_model(grid).show())) is Normalize
    ax = make_model(grid).show(vmin=1450.0, vmax=1550.0)
    assert bounds(shown_norm(ax)) == (1450.0, 1550.0)


def test_a_linear_scale_spans_the_data_but_a_two_slope_one_does_not():
    data = [1450.0, 1700.0]
    assert bounds(profile_velocity_norm("homogeneous", data=data)) == (1450.0, 1700.0)
    assert bounds(profile_velocity_norm("homogeneous", vmax=1800.0, data=data)) == (
        1450.0,
        1800.0,
    )
    assert bounds(profile_velocity_norm("shepp_logan", data=data)) == (
        1400.0,
        1600.0,
        3000.0,
    )


def test_an_explicit_norm_overrides_the_profile(grid):
    norm = Normalize(1450.0, 1550.0)
    ax = make_model(grid, "shepp_logan", scale=SCALE).show(norm=norm, vcenter=1500.0)
    assert shown_norm(ax) is norm
