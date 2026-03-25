"""MaldiAMRKit -- MALDI-TOF preprocessing toolkit for AMR prediction.

Subpackage guide
----------------
- ``maldiamrkit.builder``        -- build_driams_dataset, ProcessingHandler, BuildReport
- ``maldiamrkit.filters``        -- SpeciesFilter, DrugFilter, QualityFilter, MetadataFilter
- ``maldiamrkit.preprocessing``  -- PreprocessingPipeline, transformers, binning, quality, merging
- ``maldiamrkit.alignment``      -- Warping, RawWarping
- ``maldiamrkit.detection``      -- MaldiPeakDetector
- ``maldiamrkit.evaluation``     -- AMR metrics, splitting, LabelEncoder
- ``maldiamrkit.visualization``  -- plot_spectrum, plot_pseudogel, plot_peaks, plot_alignment
- ``maldiamrkit.loader``          -- load_driams_dataset
- ``maldiamrkit.io``             -- read_spectrum

Examples
--------
>>> from maldiamrkit import MaldiSpectrum, MaldiSet
>>> from maldiamrkit.preprocessing import PreprocessingPipeline
>>> from maldiamrkit.alignment import Warping
>>> from maldiamrkit import build_driams_dataset, load_driams_dataset, ProcessingHandler
"""

from .builder import BuildReport, ProcessingHandler, build_driams_dataset
from .dataset import MaldiSet
from .loader import load_driams_dataset
from .spectrum import MaldiSpectrum

__version__ = "0.9.0"
__author__ = "Ettore Rocchi"

__all__ = [
    "BuildReport",
    "MaldiSet",
    "MaldiSpectrum",
    "ProcessingHandler",
    "build_driams_dataset",
    "load_driams_dataset",
    "__version__",
    "__author__",
]
