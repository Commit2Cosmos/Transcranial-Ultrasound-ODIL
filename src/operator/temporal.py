"""Temporal finite-difference operators(u_tt)"""

from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .base import DenseOperator


class TemporalOperator(DenseOperator):
    """Base class for time stencil operators."""

    @abstractmethod
    def apply(self, u: jax.Array, init_ut: jax.Array | None = None, **kwargs) -> jax.Array:
        """Return dt^2 * u_tt on the full (nt, nx, ny) field."""
        pass


@dataclass
class TimeOperator2ndOrder(TemporalOperator):
    """2nd-order time stencil."""
    pass

@dataclass
class TimeOperator4thOrder(TemporalOperator):
    """4th-order time stencil."""
    pass