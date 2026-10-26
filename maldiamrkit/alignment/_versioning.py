"""Version stamp for pickled aligners.

A pickled :class:`~maldiamrkit.alignment.Warping` or
:class:`~maldiamrkit.alignment.RawWarping` records the maldiamrkit version it
was saved with. An aligner fitted with maldiamrkit 0.21 or earlier lacks the
parameters added in 0.22 and aligns differently, so it is refused instead of
being completed with defaults (which would align new spectra differently
from the training data).
"""

from __future__ import annotations

import re
import warnings

_PARAMS_SINCE_0_22 = (
    "shift_estimator",
    "min_gain",
    "min_support",
    "n_peaks",
    "prealign",
    "renormalize",
)


def _current_version() -> str:
    import maldiamrkit

    return maldiamrkit.__version__


def _version_tuple(text: object) -> tuple[int, int, int] | None:
    """Integers of the first three dotted fields of a version, or None."""
    if not isinstance(text, str):
        return None
    match = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _versioned_state(state: dict) -> dict:
    """Stamp the pickled state with the running maldiamrkit version."""
    return dict(state, _maldiamrkit_version=_current_version())


def _checked_state(name: str, state: dict) -> dict:
    """Return the state to restore, warning when it comes from an older release."""
    state = dict(state)
    version = state.pop("_maldiamrkit_version", None)
    current = _current_version()
    if any(param not in state for param in _PARAMS_SINCE_0_22):
        message = (
            f"This {name} was fitted with maldiamrkit 0.21 or earlier, whose "
            f"alignment differs from maldiamrkit {current}: it must be refitted "
            "(fit, transform and get_alignment_quality raise), or used with "
            "maldiamrkit < 0.22."
        )
    elif version is None:
        message = (
            f"This {name} was pickled with maldiamrkit 0.22.0 or earlier, which "
            f"did not record its version; it is loaded with maldiamrkit "
            f"{current} and can still be used."
        )
    else:
        saved, running = _version_tuple(version), _version_tuple(current)
        if saved is not None and (running is None or saved >= running):
            return state
        message = (
            f"This {name} was pickled with maldiamrkit {version} and is loaded "
            f"with maldiamrkit {current}."
        )
    warnings.warn(message, UserWarning, stacklevel=3)
    return state


def _check_supported(estimator: object) -> None:
    """Raise when ``estimator`` holds the state of maldiamrkit 0.21 or earlier."""
    if all(hasattr(estimator, param) for param in _PARAMS_SINCE_0_22):
        return
    name = type(estimator).__name__
    raise RuntimeError(
        f"This {name} was fitted with maldiamrkit 0.21 or earlier, whose "
        "alignment differs from maldiamrkit 0.22: create and fit a new "
        f"{name} with the current version (or keep maldiamrkit < 0.22 to use "
        "this one)."
    )
