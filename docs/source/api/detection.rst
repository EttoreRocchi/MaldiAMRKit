Detection Module
================

Peak detection algorithms and transformers.

``MaldiPeakDetector`` supports parallel processing via the ``n_jobs`` parameter.
Use ``n_jobs=-1`` to utilize all available CPU cores.

MaldiPeakDetector
-----------------

.. autoclass:: maldiamrkit.detection.peak_detector.MaldiPeakDetector
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Processing Example
---------------------------

.. code-block:: python

    from maldiamrkit import MaldiPeakDetector

    # Parallel peak detection
    detector = MaldiPeakDetector(
        method="local",
        prominence=0.01,
        n_jobs=-1  # use all cores
    )
    peaks = detector.fit_transform(X)
