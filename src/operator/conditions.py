"Boundary and Initial conditions for finite-difference stencils"

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class Conditions(ABC):
    """Base class for constraint helpers."""

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError
    
    
class NeumannMirrorBC2nd(Conditions):
    """Mirror ghost neighbours for 2nd-order spatial stencils (zero normal derivative)."""
    def patch_spatial_neighbors(
        self,
        uxm: jax.Array,
        uxp: jax.Array,
        uym: jax.Array,
        uyp: jax.Array,
        utm: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        pass

    def apply(self, *args, **kwargs):
        pass


class NeumannMirrorBC4th(Conditions):
    """Mirror ghost neighbours for 4th-order spatial stencils (+/-1 and +/-2)."""
    def patch_spatial_neighbors(
        self,
        uxm2: jax.Array,
        uxm: jax.Array,
        uxp: jax.Array,
        uxp2: jax.Array,
        uym2: jax.Array,
        uym: jax.Array,
        uyp: jax.Array,
        uyp2: jax.Array,
        utm: jax.Array,
    ) -> tuple[jax.Array, ...]:
        pass

    def apply(self, *args, **kwargs):
        pass


class InitialConditions(Conditions):
    """Hard displacement IC enforced as a residual row at t=0."""
    pass