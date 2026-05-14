"""Deprecated location for :class:`LabelEncoder`.

Moved to :mod:`maldiamrkit.susceptibility` in v0.15. Attribute access on
this module emits a :class:`DeprecationWarning` and forwards to the new
location; the shim will be removed in a future release.
"""

from __future__ import annotations

import warnings

_DEPRECATED = ("LabelEncoder", "IntermediateHandling")


def __getattr__(name: str):
    if name in _DEPRECATED:
        warnings.warn(
            "maldiamrkit.evaluation.label_encoder is deprecated; "
            "import from maldiamrkit.susceptibility instead. "
            "Will be removed in v0.17.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..susceptibility.label_encoder import (
            IntermediateHandling,
            LabelEncoder,
        )

        return {
            "LabelEncoder": LabelEncoder,
            "IntermediateHandling": IntermediateHandling,
        }[name]
    raise AttributeError(
        f"module 'maldiamrkit.evaluation.label_encoder' has no attribute {name!r}"
    )


def __dir__() -> list[str]:
    return list(_DEPRECATED)


__all__ = list(_DEPRECATED)
