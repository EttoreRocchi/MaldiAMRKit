Core Module
===========

Core data structures for MALDI-TOF mass spectrometry analysis.

MaldiSpectrum
-------------

.. autoclass:: maldiamrkit.core.spectrum.MaldiSpectrum
   :members:
   :undoc-members:
   :show-inheritance:

MaldiSet
--------

.. autoclass:: maldiamrkit.core.dataset.MaldiSet
   :members:
   :undoc-members:
   :show-inheritance:

``MaldiSet.from_directory()`` supports parallel loading via the ``n_jobs`` parameter:

.. code-block:: python

    from maldiamrkit import MaldiSet

    # Parallel loading (use all cores)
    data = MaldiSet.from_directory(
        "spectra/",
        "metadata.csv",
        n_jobs=-1
    )

PreprocessingSettings
---------------------

.. autoclass:: maldiamrkit.core.config.PreprocessingSettings
   :members:
   :undoc-members:
   :show-inheritance:
