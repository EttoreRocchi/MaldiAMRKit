"""Data building and loading utilities.

Submodules
----------
- ``input_layouts`` -- :class:`InputLayout` adapters for :class:`DatasetBuilder`
- ``dataset_layouts`` -- :class:`DatasetLayout` adapters for :class:`DatasetLoader`
- ``builder`` -- :class:`DatasetBuilder`, :class:`ProcessingHandler`, :class:`BuildReport`
- ``loader`` -- :class:`DatasetLoader`
"""

from .builder import BuildReport, DatasetBuilder, ProcessingHandler
from .dataset_layouts import DatasetLayout, DRIAMSLayout, MARISMaLayout
from .duplicates import DuplicateStrategy
from .input_layouts import BrukerTreeLayout, FlatLayout, InputLayout
from .loader import DatasetLoader

__all__ = [
    # Duplicate handling
    "DuplicateStrategy",
    # Input layouts (for DatasetBuilder)
    "InputLayout",
    "FlatLayout",
    "BrukerTreeLayout",
    # Dataset layouts (for DatasetLoader)
    "DatasetLayout",
    "DRIAMSLayout",
    "MARISMaLayout",
    # Builder
    "DatasetBuilder",
    "ProcessingHandler",
    "BuildReport",
    # Loader
    "DatasetLoader",
]
