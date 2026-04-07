"""MaldiAMRKit -- MALDI-TOF preprocessing toolkit for AMR prediction.

Subpackage guide
----------------
- ``maldiamrkit.data``           -- DatasetBuilder, DatasetLoader, InputLayout, DatasetLayout
- ``maldiamrkit.filters``        -- SpeciesFilter, DrugFilter, QualityFilter, MetadataFilter
- ``maldiamrkit.preprocessing``  -- PreprocessingPipeline, transformers, binning, quality, merging
- ``maldiamrkit.alignment``      -- Warping, RawWarping, create_raw_input, AlignmentStrategy
- ``maldiamrkit.detection``      -- MaldiPeakDetector
- ``maldiamrkit.evaluation``     -- AMR metrics, splitting, LabelEncoder
- ``maldiamrkit.visualization``  -- plot_spectrum, plot_pseudogel, plot_peaks, plot_alignment, plot_pca, plot_tsne, plot_umap
- ``maldiamrkit.io``             -- read_spectrum, parse_mic_column

Examples
--------
>>> from maldiamrkit import MaldiSpectrum, MaldiSet
>>> from maldiamrkit.preprocessing import PreprocessingPipeline
>>> from maldiamrkit.alignment import Warping
>>> from maldiamrkit.data import DatasetBuilder, DatasetLoader, FlatLayout, DRIAMSLayout
"""

from .dataset import MaldiSet
from .spectrum import MaldiSpectrum

__version__ = "0.11.2"
__author__ = "Ettore Rocchi"

__all__ = [
    "MaldiSet",
    "MaldiSpectrum",
    "__version__",
    "__author__",
]
