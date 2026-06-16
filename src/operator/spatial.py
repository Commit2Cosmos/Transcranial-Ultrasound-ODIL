"""Spatial finite-difference operators (Laplacian)."""

from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .base import DenseOperator
from .conditions import NeumannMirrorBC2nd, NeumannMirrorBC4th

# second-order 5 point Laplacian
def _laplacian_5pt(
    utm: jax.Array,
    uxm: jax.Array,
    uxp: jax.Array,
    uym: jax.Array,
    uyp: jax.Array,
) -> jax.Array:
    return (uxm - 2.0 * utm + uxp) + (uym - 2.0 * utm + uyp)

# fourth-order uxx or uyy
def _fourth_derivative_1d(
    um2: jax.Array,
    um1: jax.Array,
    u: jax.Array,
    up1: jax.Array,
    up2: jax.Array,
) -> jax.Array:
    # 4th-order central u_xx: (-u_{-2} + 16*u_{-1} - 30*u + 16*u_{+1} - u_{+2}) / 12
    return (-um2 + 16.0 * um1 - 30.0 * u + 16.0 * up1 - up2) / 12.0


class SpatialOperator(DenseOperator):
    """Spatial Laplacian on the t-1 field, cell-centred grid."""

    @abstractmethod
    def gather_neighbors(self, utm: jax.Array) -> tuple[jax.Array, ...]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, utm: jax.Array, bc=None, **kwargs) -> jax.Array:
        """Discrete Laplacian (u_xx + u_yy) on utm."""
        raise NotImplementedError


@dataclass
class Laplacian2ndOrder(SpatialOperator):
    """2nd-order 5-point Laplacian."""

    # collects the four spatial neighbors of the t-1 field
    def gather_neighbors(
        self, utm: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        uxm = jnp.roll(utm, 1, axis=1) # neigbor to the left
        uxp = jnp.roll(utm, -1, axis=1) # neigbor to the right
        uym = jnp.roll(utm, 1, axis=2) # neigbor below
        uyp = jnp.roll(utm, -1, axis=2) # neigbor above
        return uxm, uxp, uym, uyp

    def apply(self, utm: jax.Array, bc: NeumannMirrorBC2nd | None = None) -> jax.Array:
        bc = bc or NeumannMirrorBC2nd()
        uxm, uxp, uym, uyp = self.gather_neighbors(utm) # collect the neigbors
        uxm, uxp, uym, uyp = bc.patch_spatial_neighbors(uxm, uxp, uym, uyp, utm) # replace teh wrong periodic roll with the mirrored interior values
        return _laplacian_5pt(utm, uxm, uxp, uym, uyp)


@dataclass
class Laplacian4thOrder(SpatialOperator):
    """4th-order 9-point Laplacian."""

    def gather_neighbors(self, utm: jax.Array) -> tuple[jax.Array, ...]:
        uxm2 = jnp.roll(utm, 2, axis=1) # two cells left(i-2,j)
        uxm = jnp.roll(utm, 1, axis=1) # one cell left(i-1,j)
        uxp = jnp.roll(utm, -1, axis=1) # one cell right(i+1,j)
        uxp2 = jnp.roll(utm, -2, axis=1) # two cells right(i+2,j)
        uym2 = jnp.roll(utm, 2, axis=2) # two cells down(i,j-2)
        uym = jnp.roll(utm, 1, axis=2) # one cell down(i,j-1)
        uyp = jnp.roll(utm, -1, axis=2) # one cell up(i,j+1)
        uyp2 = jnp.roll(utm, -2, axis=2)    
        return uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2

    def apply(self, utm: jax.Array, bc: NeumannMirrorBC4th | None = None) -> jax.Array:
        bc = bc or NeumannMirrorBC4th()
        uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2 = self.gather_neighbors(utm)
        neighbours = bc.patch_spatial_neighbors(
            uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2, utm
        )
        uxm2, uxm, uxp, uxp2, uym2, uym, uyp, uyp2 = neighbours
        u_xx = _fourth_derivative_1d(uxm2, uxm, utm, uxp, uxp2)
        u_yy = _fourth_derivative_1d(uym2, uym, utm, uyp, uyp2)
        return u_xx + u_yy