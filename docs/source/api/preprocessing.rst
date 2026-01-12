Preprocessing Module
====================

Functions for preprocessing MALDI-TOF spectra.

Pipeline
--------

.. autofunction:: maldiamrkit.preprocessing.pipeline.preprocess

Binning
-------

.. autofunction:: maldiamrkit.preprocessing.binning.bin_spectrum

.. autofunction:: maldiamrkit.preprocessing.binning.get_bin_metadata

Binning Methods
~~~~~~~~~~~~~~~

MaldiAMRKit supports multiple binning strategies:

**Uniform** (default): Fixed-width bins across the m/z range.

.. code-block:: python

    spec.bin(bin_width=3)  # 3 Da bins

**Logarithmic**: Bin width scales with m/z, matching instrument resolution.

.. code-block:: python

    spec.bin(bin_width=3, method="logarithmic")

**Adaptive**: Smaller bins in peak-dense regions, larger bins elsewhere.

.. code-block:: python

    spec.bin(method="adaptive", adaptive_min_width=1.0, adaptive_max_width=10.0)

**Custom**: User-defined bin edges for domain-specific analysis.

.. code-block:: python

    spec.bin(method="custom", custom_edges=[2000, 5000, 10000, 15000, 20000])

Bin metadata is available via the ``bin_metadata`` attribute:

.. code-block:: python

    print(spec.bin_metadata.head())
    #    bin_index  bin_start  bin_end  bin_width
    # 0          0     2000.0   2003.0        3.0

Quality Metrics
---------------

.. autofunction:: maldiamrkit.preprocessing.quality.estimate_snr
