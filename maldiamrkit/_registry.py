"""Shared machinery for the package's name-dispatched component registries.

The public registration APIs (``register_spectral_metric``,
``register_transformer``, ``register_binning_method`` and their
``unregister_*`` / ``list_*`` companions) are thin wrappers around one
:class:`_ComponentRegistry` instance per component kind, so the three
registries share a single implementation of the built-in protection,
override, restore, and name-resolution semantics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class _ComponentRegistry(dict):
    """A ``name -> component`` mapping whose built-in entries are protected.

    Behaves as a plain dict for lookups, while the mutation methods enforce
    the registry contract:

    - registering a new name is always allowed (re-registering a custom name
      replaces it silently);
    - replacing a built-in requires ``override=True``;
    - unregistering an overridden built-in restores the default component;
    - unregistering a pristine built-in is refused.

    Parameters
    ----------
    kind : str
        Full component noun used in error messages, e.g. ``"spectral
        metric"``.
    short : str
        Short noun used where the messages already say the kind, e.g.
        ``"metric"``.
    register_fn : str
        Name of the public registration function to point users at.
    entries : dict
        The built-in components. A copy is kept as the immutable set of
        defaults that :meth:`unregister` can restore.
    validate : callable, optional
        Called with the candidate component before registration; should
        raise ``TypeError`` for unacceptable components.
    """

    def __init__(
        self,
        kind: str,
        short: str,
        register_fn: str,
        entries: dict[str, Any],
        validate: Callable[[Any], None] | None = None,
    ) -> None:
        super().__init__(entries)
        self._kind = kind
        self._short = short
        self._register_fn = register_fn
        self._validate = validate
        self._defaults: dict[str, Any] = dict(entries)

    @property
    def defaults(self) -> dict[str, Any]:
        """The built-in components (``name -> component``), never mutated."""
        return self._defaults

    def register(self, name: str, component: Any, *, override: bool = False) -> None:
        if self._validate is not None:
            self._validate(component)
        if name in self._defaults and not override:
            raise ValueError(
                f"{name!r} is a built-in {self._kind}. "
                "Pass override=True to replace it."
            )
        self[name] = component

    def unregister(self, name: str) -> None:
        if name in self._defaults:
            default = self._defaults[name]
            if self.get(name) is default:
                raise ValueError(
                    f"Cannot unregister built-in {self._kind} {name!r}. "
                    f"Built-in {self._short}s are {sorted(self._defaults)}."
                )
            self[name] = default
            return
        try:
            del self[name]
        except KeyError:
            raise KeyError(f"No {self._kind} named {name!r} is registered.") from None

    def names(self) -> list[str]:
        return sorted(self)

    def resolve_key(
        self,
        value: Any,
        enum_cls: type[Enum] | None = None,
        *,
        not_a: str = "valid",
    ) -> str:
        """Return the registry key for *value*, an enum member or a name.

        Raises
        ------
        ValueError
            If *value* names no registered component. The message lists the
            registered names and points at the registration function.
        """
        if enum_cls is not None and isinstance(value, enum_cls):
            key = value.value
        else:
            key = value
        try:
            known = key in self
        except TypeError:
            known = False
        if not known:
            raise ValueError(
                f"{value!r} is not a {not_a} {self._kind}. Use one of "
                f"{self.names()} or register a custom {self._short} with "
                f"{self._register_fn}()."
            )
        return key
