"""Shared helpers for the visualization subpackage.

Centralises the small utilities that every plot module needs so we
don't drift apart: the ``show=True`` + non-interactive-backend warning
handshake, the susceptibility-aware label ordering, and the default
R/I/S display-name mapping.
"""

from __future__ import annotations

import warnings
from typing import Any

_SUSCEPTIBILITY_ORDER: dict[str, int] = {"S": 0, "I": 1, "R": 2}

DEFAULT_LABEL_MAP: dict = {
    0: "Susceptible (S)",
    1: "Resistant (R)",
    "S": "Susceptible (S)",
    "I": "Intermediate (I)",
    "R": "Resistant (R)",
}


def show_with_warning(show: bool) -> None:
    """Call ``plt.show()`` with a backend-compatibility warning."""
    import matplotlib
    import matplotlib.pyplot as plt

    if show:
        if not matplotlib.is_interactive():
            warnings.warn(
                "matplotlib is using a non-interactive backend; "
                "plt.show() may not display the figure",
                UserWarning,
                stacklevel=3,
            )
        plt.show()


def order_labels(labels: list[Any]) -> list[Any]:
    """Return labels ordered S → I → R, then numeric ascending, then alphabetically.

    Leaves the input unchanged when no label matches the S/I/R set,
    preserving insertion order for non-susceptibility data.
    """

    def key(lab):
        s = str(lab).strip().upper()
        if s in _SUSCEPTIBILITY_ORDER:
            return (0, _SUSCEPTIBILITY_ORDER[s], s)
        try:
            return (1, float(lab), s)
        except (TypeError, ValueError):
            return (2, 0, s)

    return sorted(labels, key=key)


def resolve_display_label(label: Any, label_map: dict | None = None) -> str:
    """Map a raw label (0/1/R/I/S/...) to its display string.

    Caller-supplied ``label_map`` takes precedence over the default.
    """
    if label_map and label in label_map:
        return str(label_map[label])
    return str(DEFAULT_LABEL_MAP.get(label, label))
