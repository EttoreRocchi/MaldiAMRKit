"""Data building and loading utilities.

Submodules
----------
- ``input_layouts`` -- :class:`InputLayout` adapters for :class:`DatasetBuilder`
- ``dataset_layouts`` -- :class:`DatasetLayout` adapters for :class:`DatasetLoader`
- ``builder`` -- :class:`DatasetBuilder`, :class:`ProcessingHandler`, :class:`BuildReport`
- ``loader`` -- :class:`DatasetLoader`
- ``site_info`` -- :class:`SiteInfo`, :class:`BuildInfo`, ``read_site_info``,
  ``write_site_info`` (self-describing dataset manifest written at build time
  and consulted by :class:`DRIAMSLayout` at load time)
"""

from .builder import BuildReport, DatasetBuilder, ProcessingHandler
from .dataset_layouts import DatasetLayout, DRIAMSLayout, MARISMaLayout
from .duplicates import DuplicateStrategy
from .input_layouts import BrukerTreeLayout, FlatLayout, InputLayout
from .loader import DatasetLoader
from .site_info import (
    CURRENT_FORMAT_VERSION,
    MANIFEST_FILENAME,
    BuildInfo,
    SiteInfo,
    read_site_info,
    write_site_info,
)

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
    # Dataset manifest
    "SiteInfo",
    "BuildInfo",
    "read_site_info",
    "write_site_info",
    "MANIFEST_FILENAME",
    "CURRENT_FORMAT_VERSION",
]
