"""MaldiAMRKit -- MALDI-TOF preprocessing toolkit for AMR prediction.

Subpackage guide
----------------
- ``maldiamrkit.filters``        -- SpeciesFilter, DrugFilter, QualityFilter, MetadataFilter
- ``maldiamrkit.preprocessing``  -- PreprocessingPipeline, transformers, binning, quality, merging
- ``maldiamrkit.alignment``      -- Warping, RawWarping
- ``maldiamrkit.detection``      -- MaldiPeakDetector
- ``maldiamrkit.evaluation``     -- AMR metrics, splitting, LabelEncoder
- ``maldiamrkit.io``             -- read_spectrum

Examples
--------
>>> from maldiamrkit import MaldiSpectrum, MaldiSet
>>> from maldiamrkit.preprocessing import PreprocessingPipeline
>>> from maldiamrkit.alignment import Warping
"""

from .dataset import MaldiSet
from .spectrum import MaldiSpectrum

__version__ = "0.7.0"
__author__ = "Ettore Rocchi"

__all__ = [
    "MaldiSpectrum",
    "MaldiSet",
    "__version__",
    "__author__",
]
