"""Read a library default from the signature that owns it (ADR 0001).

The run config (``odil_wave.experiment.config``) and outer constructors that
thread a parameter through to its consumer read the default here rather than
repeat the literal. Entry point: :func:`library_default`.
"""

import inspect
from typing import Any, Callable


def library_default(owner: Callable, name: str) -> Any:
    """The default of parameter ``name`` in the signature of class or function
    ``owner``. Raises ``TypeError`` when ``name`` is required there."""
    params = inspect.signature(owner).parameters
    if name not in params:
        raise TypeError(f"{owner.__qualname__} has no parameter {name!r}")
    default = params[name].default
    if default is inspect.Parameter.empty:
        raise TypeError(
            f"{owner.__qualname__}({name}=...) is required: it has no library default"
        )
    return default
