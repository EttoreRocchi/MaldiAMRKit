Alignment Module
================

Spectral alignment and warping transformers.

Both ``Warping`` and ``RawWarping`` support parallel processing via the ``n_jobs`` parameter.
Use ``n_jobs=-1`` to utilize all available CPU cores.

Warping (Binned Spectra)
------------------------

.. autoclass:: maldiamrkit.alignment.warping.Warping
   :members:
   :undoc-members:
   :show-inheritance:

RawWarping (Full Resolution)
----------------------------

.. autoclass:: maldiamrkit.alignment.raw_warping.RawWarping
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: maldiamrkit.alignment.raw_warping.create_raw_input

Example Usage
-------------

.. code-block:: python

    from maldiamrkit import Warping, RawWarping, create_raw_input
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Alignment on binned data
    warper = Warping(method="piecewise", n_jobs=-1)
    X_aligned = warper.fit_transform(X_binned)

    # Raw warping: create input from directory, get binned output
    X_raw = create_raw_input("spectra/")  # DataFrame with file paths
    raw_warper = RawWarping(method="piecewise", bin_width=3, n_jobs=-1)
    X_binned = raw_warper.fit_transform(X_raw)

    # Use in sklearn pipeline
    pipe = Pipeline([
        ("warp", RawWarping(method="piecewise", bin_width=3)),
        ("scaler", StandardScaler()),
    ])
    X_processed = pipe.fit_transform(X_raw)
