"""Composable preprocessing pipeline for MALDI-TOF spectra.

Similar to :class:`sklearn.pipeline.Pipeline` but designed for spectrum
DataFrames with ``mass`` and ``intensity`` columns.

Examples
--------
>>> from maldiamrkit.preprocessing import PreprocessingPipeline
>>> from maldiamrkit.preprocessing.transformers import *
>>>
>>> # Default pipeline (standard preprocessing)
>>> pipe = PreprocessingPipeline.default()
>>> preprocessed = pipe(raw_df)
>>>
>>> # Custom pipeline
>>> pipe = PreprocessingPipeline([
...     ("clip", ClipNegatives()),
...     ("log", LogTransform()),
...     ("smooth", SavitzkyGolaySmooth(window_length=15)),
...     ("baseline", SNIPBaseline(half_window=30)),
...     ("trim", MzTrimmer(mz_min=2000, mz_max=20000)),
...     ("norm", TICNormalizer()),
... ])
>>> preprocessed = pipe(raw_df)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

from .transformers import (
    _TRANSFORMER_REGISTRY,
    ClipNegatives,
    MzTrimmer,
    PreprocessingStep,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    SqrtTransform,
    TICNormalizer,
)

__all__ = ["PreprocessingPipeline", "PreprocessingStep"]


def _external_stacklevel() -> int:
    """Stacklevel for ``warnings.warn`` pointing outside ``maldiamrkit``.

    Walks the stack from the caller outwards until the first frame that does
    not belong to this package, so warnings are attributed to the user's
    call site whether ``to_dict`` is reached directly or via ``to_json`` /
    ``to_yaml`` / a :class:`~maldiamrkit.data.builder.ProcessingHandler`.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(pkg_dir):
        frame = frame.f_back
        level += 1
    return level


class PreprocessingPipeline:
    """Composable pipeline of preprocessing steps for MALDI-TOF spectra.

    Parameters
    ----------
    steps : list of (str, transformer) tuples
        Named preprocessing steps. Each transformer must be callable,
        accepting and returning a ``pd.DataFrame`` with ``mass`` and
        ``intensity`` columns.

    Examples
    --------
    >>> pipe = PreprocessingPipeline.default()
    >>> preprocessed = pipe(raw_spectrum_df)
    """

    def __init__(self, steps: list[tuple[str, PreprocessingStep]]):
        self.steps = list(steps)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps sequentially.

        Parameters
        ----------
        df : pd.DataFrame
            Raw spectrum with ``mass`` and ``intensity`` columns.

        Returns
        -------
        pd.DataFrame
            Preprocessed spectrum.
        """
        for _name, step in self.steps:
            df = step(df)
        return df

    @classmethod
    def default(cls) -> PreprocessingPipeline:
        """Return the standard preprocessing pipeline.

        Steps: clip negatives -> sqrt transform -> Savitzky-Golay smoothing ->
        SNIP baseline -> m/z trim (2000-20000 Da) -> TIC normalization.

        Returns
        -------
        PreprocessingPipeline
            Default pipeline instance.
        """
        return cls(
            [
                ("clip", ClipNegatives()),
                ("sqrt", SqrtTransform()),
                ("smooth", SavitzkyGolaySmooth(window_length=21, polyorder=2)),
                ("baseline", SNIPBaseline(half_window=40)),
                ("trim", MzTrimmer(mz_min=2000, mz_max=20000)),
                ("normalize", TICNormalizer()),
            ]
        )

    def get_step(self, name: str) -> object:
        """Get a step by name.

        Parameters
        ----------
        name : str
            Step name.

        Returns
        -------
        object
            The transformer associated with that name.

        Raises
        ------
        KeyError
            If no step with that name exists.
        """
        for step_name, step in self.steps:
            if step_name == name:
                return step
        raise KeyError(f"Step '{name}' not found. Available: {self.step_names}")

    @property
    def step_names(self) -> list[str]:
        """Return the names of all steps."""
        return [name for name, _ in self.steps]

    @property
    def mz_range(self) -> tuple[int, int]:
        """Extract (mz_min, mz_max) from the MzTrimmer step.

        Returns
        -------
        tuple[int, int]
            The m/z range from the MzTrimmer step, or the default
            (2000, 20000) if no MzTrimmer is present.
        """
        for _, step in self.steps:
            if isinstance(step, MzTrimmer):
                return step.mz_min, step.mz_max
        return 2000, 20000

    def to_dict(self) -> dict:
        """Serialize the pipeline to a dictionary.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON/YAML serialization.

        Warns
        -----
        UserWarning
            If a step serializes under a name that is not registered (or
            registered to a different class), so the resulting config could
            not be rebuilt faithfully by :meth:`from_dict`. Register the
            transformer with
            :func:`~maldiamrkit.preprocessing.register_transformer` to make
            the pipeline round-trip.
        """
        steps = []
        for name, step in self.steps:
            payload = step.to_dict()
            self._warn_if_unregistered(name, step, payload)
            steps.append({"step_name": name, **payload})
        return {"steps": steps}

    @staticmethod
    def _warn_if_unregistered(
        step_name: str,
        step: PreprocessingStep,
        payload: dict,
    ) -> None:
        """Warn when a serialized step could not be rebuilt by ``from_dict``."""
        transformer_name = payload.get("name")
        if transformer_name is None:
            warnings.warn(
                f"Step {step_name!r} serialises without a 'name' key, so "
                "PreprocessingPipeline.from_dict() cannot rebuild it. Its "
                "to_dict() should return {'name': <registered name>, ...}.",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
            return
        registered = _TRANSFORMER_REGISTRY.get(transformer_name)
        if registered is None:
            warnings.warn(
                f"Step {step_name!r} serialises under name "
                f"{transformer_name!r}, which is not registered, so "
                "PreprocessingPipeline.from_dict() cannot rebuild it. Call "
                f"register_transformer({transformer_name!r}, ...) before "
                "loading this config.",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
        elif registered is not type(step):
            warnings.warn(
                f"Step {step_name!r} of class {type(step).__name__} "
                f"serialises under name {transformer_name!r}, which is "
                f"registered to {registered.__name__}, so "
                "PreprocessingPipeline.from_dict() would rebuild a different "
                "class. Register this class under a name of its own.",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessingPipeline:
        """Reconstruct a pipeline from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        PreprocessingPipeline
            Reconstructed pipeline.

        Raises
        ------
        ValueError
            If the config is missing the ``'steps'`` key, a step is missing
            its ``'step_name'`` or ``'name'`` key, or a step names a
            transformer that is not registered. Custom transformers must be
            registered with
            :func:`~maldiamrkit.preprocessing.register_transformer` before
            the config is loaded.
        """
        try:
            step_dicts = d["steps"]
        except KeyError:
            raise ValueError("Pipeline config is missing the 'steps' key.") from None
        steps = []
        for step_dict in step_dicts:
            try:
                step_name = step_dict["step_name"]
            except KeyError:
                raise ValueError(
                    f"Pipeline step {step_dict!r} is missing the 'step_name' key."
                ) from None
            try:
                transformer_name = step_dict["name"]
            except KeyError:
                raise ValueError(
                    f"Step {step_name!r} is missing the 'name' key that "
                    "identifies its transformer. A step's to_dict() must "
                    "return {'name': <registered name>, ...}."
                ) from None
            transformer_cls = _TRANSFORMER_REGISTRY[
                _TRANSFORMER_REGISTRY.resolve_key(transformer_name, not_a="registered")
            ]

            # Extract constructor kwargs (everything except step_name and name)
            kwargs = {
                k: v for k, v in step_dict.items() if k not in ("step_name", "name")
            }
            steps.append((step_name, transformer_cls(**kwargs)))

        return cls(steps)

    def to_json(self, path: str | Path) -> None:
        """Save the pipeline configuration to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> PreprocessingPipeline:
        """Load a pipeline from a JSON file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        PreprocessingPipeline
            Reconstructed pipeline.
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_yaml(self, path: str | Path) -> None:
        """Save the pipeline configuration to a YAML file.

        Requires ``pyyaml`` to be installed.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessingPipeline:
        """Load a pipeline from a YAML file.

        Requires ``pyyaml`` to be installed.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        PreprocessingPipeline
            Reconstructed pipeline.
        """
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def __repr__(self) -> str:
        steps_repr = ",\n  ".join(f"('{name}', {step!r})" for name, step in self.steps)
        return f"PreprocessingPipeline([\n  {steps_repr}\n])"

    def __len__(self) -> int:
        return len(self.steps)
