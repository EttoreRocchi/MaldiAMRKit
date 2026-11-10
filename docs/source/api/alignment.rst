Alignment module
================

Spectral alignment and warping transformers.

Both ``Warping`` and ``RawWarping`` support parallel processing via the ``n_jobs`` parameter.
Use ``n_jobs=-1`` to utilise all available CPU cores.

Warping (binned spectra)
------------------------

.. autoclass:: maldiamrkit.alignment.Warping
   :members:
   :undoc-members:
   :show-inheritance:

RawWarping (full resolution)
----------------------------

.. autoclass:: maldiamrkit.alignment.RawWarping
   :members:
   :undoc-members:
   :show-inheritance:

Alignment strategies
--------------------

.. autoclass:: maldiamrkit.alignment.AlignmentStrategy
   :members:

Alignment methods
-----------------

.. autoclass:: maldiamrkit.alignment.AlignmentMethod
   :members:
   :undoc-members:
   :show-inheritance:

Utility functions
-----------------

.. autofunction:: maldiamrkit.alignment.create_raw_input

.. autofunction:: maldiamrkit.alignment.align_peaks

Example usage
-------------

.. code-block:: python

    from maldiamrkit.alignment import Warping, RawWarping, create_raw_input
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Alignment on binned data
    warper = Warping(method="lowess", n_jobs=-1)
    X_aligned = warper.fit_transform(X_binned)

    # Raw warping: create input from directory, get binned output
    X_raw = create_raw_input("spectra/")  # DataFrame with file paths
    raw_warper = RawWarping(method="lowess", bin_width=3, n_jobs=-1)
    X_binned = raw_warper.fit_transform(X_raw)

    # Use in sklearn pipeline
    pipe = Pipeline([
        ("warp", RawWarping(method="lowess", bin_width=3)),
        ("scaler", StandardScaler()),
    ])
    X_processed = pipe.fit_transform(X_raw)
