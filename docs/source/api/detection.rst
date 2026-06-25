Detection Module
================

Peak detection algorithms and transformers.

``MaldiPeakDetector`` supports parallel processing via the ``n_jobs`` parameter.
Use ``n_jobs=-1`` to utilize all available CPU cores.

MaldiPeakDetector
-----------------

.. autoclass:: maldiamrkit.detection.MaldiPeakDetector
   :members:
   :undoc-members:
   :show-inheritance:

Peak Detection Methods
----------------------

.. autoclass:: maldiamrkit.detection.PeakMethod
   :members:
   :undoc-members:
   :show-inheritance:

Peak Sets
---------

``PeakSet`` / ``PeakList`` represent spectra as variable-length
``(m/z, intensity)`` peak sets. ``MaldiPeakDetector.transform_peaklist`` and
``create_peakset_input`` build them.

Peak extraction is a pure per-spectrum function, so a ``PeakList`` precomputed
over a whole dataset is identical to one computed inside a single
cross-validation fold; caching it (via ``cache_dir=``) cannot leak. Any fitted
alignment (see :func:`maldiamrkit.alignment.align_peaks`) is applied downstream,
on the training fold's reference.

.. autoclass:: maldiamrkit.detection.PeakSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.detection.PeakList
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: maldiamrkit.detection.create_peakset_input

Parallel Processing Example
---------------------------

.. code-block:: python

    from maldiamrkit.detection import MaldiPeakDetector

    # Parallel peak detection
    detector = MaldiPeakDetector(
        method="local",
        prominence=0.01,
        n_jobs=-1  # use all cores
    )
    peaks = detector.fit_transform(X)
