API Reference
=============

This section provides detailed documentation for all public classes and
functions in MaldiAMRKit.

.. toctree::
   :maxdepth: 2

   core
   preprocessing
   alignment
   detection
   io

Summary
-------

Core Data Structures
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.MaldiSpectrum
   maldiamrkit.MaldiSet
   maldiamrkit.PreprocessingSettings

Transformers
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.Warping
   maldiamrkit.RawWarping
   maldiamrkit.MaldiPeakDetector

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldiamrkit.preprocess
   maldiamrkit.bin_spectrum
   maldiamrkit.estimate_snr
   maldiamrkit.read_spectrum
