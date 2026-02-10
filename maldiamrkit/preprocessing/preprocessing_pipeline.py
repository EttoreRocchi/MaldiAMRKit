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
from pathlib import Path

import pandas as pd

from .transformers import (
    TRANSFORMER_REGISTRY,
    ClipNegatives,
    MzTrimmer,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    SqrtTransform,
    TICNormalizer,
)


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

    def __init__(self, steps: list[tuple[str, object]]):
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

        Steps: clip negatives → sqrt transform → Savitzky-Golay smoothing →
        SNIP baseline → m/z trim (2000–20000 Da) → TIC normalization.

        Returns
        -------
        PreprocessingPipeline
            Default pipeline instance.
        """
        return cls(
            [
                ("clip", ClipNegatives()),
                ("sqrt", SqrtTransform()),
                ("smooth", SavitzkyGolaySmooth(window_length=20, polyorder=2)),
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
        """
        return {
            "steps": [
                {"step_name": name, **step.to_dict()} for name, step in self.steps
            ]
        }

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
        """
        steps = []
        for step_dict in d["steps"]:
            step_name = step_dict["step_name"]
            transformer_name = step_dict["name"]
            transformer_cls = TRANSFORMER_REGISTRY[transformer_name]

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
