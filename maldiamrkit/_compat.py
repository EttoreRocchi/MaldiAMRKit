"""Backward-compatibility utilities (internal)."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def deprecated(new_name: str, removed_in: str) -> Callable[[_F], _F]:
    """Mark a callable as deprecated.

    Parameters
    ----------
    new_name : str
        Replacement to suggest in the warning message.
    removed_in : str
        Version string where the deprecated callable will be removed.

    Returns
    -------
    Callable
        Decorator that wraps the original callable and emits a
        ``DeprecationWarning`` on every call.
    """

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{fn.__qualname__} is deprecated, use {new_name} instead. "
                f"Will be removed in v{removed_in}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
