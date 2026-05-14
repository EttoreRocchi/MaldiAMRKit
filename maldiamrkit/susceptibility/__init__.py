"""Clinical susceptibility utilities: MIC encoding, breakpoints, label encoding.

Goes from raw MIC strings to the targets needed for AMR ML pipelines:

- :class:`MICEncoder` produces ``log2(MIC)`` (regression target) plus, when
  given a :class:`BreakpointTable`, the S/I/R category and the ATU
  (Area of Technical Uncertainty) flag in a single DataFrame.
- :class:`BreakpointTable` loads clinical breakpoints from a bundled
  EUCAST version, a year, or a user-supplied YAML.
- :class:`LabelEncoder` maps R/I/S strings to binary 0/1 for downstream
  classification (previously exported from ``maldiamrkit.evaluation``).

Regression-style evaluation of continuous MIC predictions lives in
:func:`maldiamrkit.evaluation.mic_regression_report`, alongside the
binary AMR metrics it complements.

Examples
--------
>>> from maldiamrkit.susceptibility import BreakpointTable, MICEncoder
>>> bp = BreakpointTable.from_yaml("path/to/eucast_v16.yaml")  # doctest: +SKIP
>>> enc = MICEncoder(breakpoints=bp, species_col="Species", drug="Ceftriaxone")
>>> out = enc.fit_transform(df)  # doctest: +SKIP
"""

from __future__ import annotations

from .breakpoint import BreakpointResult, BreakpointTable
from .label_encoder import IntermediateHandling, LabelEncoder
from .mic_encoder import MICEncoder

__all__ = [
    "BreakpointResult",
    "BreakpointTable",
    "IntermediateHandling",
    "LabelEncoder",
    "MICEncoder",
]
