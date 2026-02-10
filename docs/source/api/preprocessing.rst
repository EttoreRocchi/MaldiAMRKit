Preprocessing Module
====================

Functions for preprocessing MALDI-TOF spectra.

PreprocessingPipeline
---------------------

.. autoclass:: maldiamrkit.preprocessing.PreprocessingPipeline
   :members:
   :undoc-members:
   :show-inheritance:

The ``preprocess()`` function is a convenience wrapper around the pipeline:

.. autofunction:: maldiamrkit.preprocessing.preprocess

Individual Transformers
-----------------------

Each transformer is a callable operating on a DataFrame with ``mass`` and
``intensity`` columns. They can be composed via :class:`PreprocessingPipeline`.

.. autoclass:: maldiamrkit.preprocessing.ClipNegatives
   :members:

.. autoclass:: maldiamrkit.preprocessing.SqrtTransform
   :members:

.. autoclass:: maldiamrkit.preprocessing.LogTransform
   :members:

.. autoclass:: maldiamrkit.preprocessing.SavitzkyGolaySmooth
   :members:

.. autoclass:: maldiamrkit.preprocessing.SNIPBaseline
   :members:

.. autoclass:: maldiamrkit.preprocessing.MzTrimmer
   :members:

.. autoclass:: maldiamrkit.preprocessing.TICNormalizer
   :members:

.. autoclass:: maldiamrkit.preprocessing.MedianNormalizer
   :members:

.. autoclass:: maldiamrkit.preprocessing.PQNNormalizer
   :members:

.. autoclass:: maldiamrkit.preprocessing.MzMultiTrimmer
   :members:

Pipeline Serialization
~~~~~~~~~~~~~~~~~~~~~~

Save and load pipeline configurations for reproducibility:

.. code-block:: python

    from maldiamrkit.preprocessing import PreprocessingPipeline

    pipe = PreprocessingPipeline.default()

    # Save to JSON
    pipe.to_json("pipeline.json")
    pipe = PreprocessingPipeline.from_json("pipeline.json")

    # Save to YAML (requires pyyaml)
    pipe.to_yaml("pipeline.yaml")
    pipe = PreprocessingPipeline.from_yaml("pipeline.yaml")

Binning
-------

.. autofunction:: maldiamrkit.preprocessing.bin_spectrum

.. autofunction:: maldiamrkit.preprocessing.get_bin_metadata

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

.. autofunction:: maldiamrkit.preprocessing.estimate_snr

.. autoclass:: maldiamrkit.preprocessing.SpectrumQuality
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.preprocessing.SpectrumQualityReport
   :no-index:
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit import MaldiSpectrum
    from maldiamrkit.preprocessing import SpectrumQuality

    # Assess spectrum quality
    spec = MaldiSpectrum("spectrum.txt").preprocess()
    qc = SpectrumQuality()  # Uses high m/z region (19500-20000) by default
    report = qc.assess(spec)

    print(f"SNR: {report.snr:.1f}")
    print(f"Peak count: {report.peak_count}")
    print(f"Total ion count: {report.total_ion_count:.2e}")
    print(f"Baseline fraction: {report.baseline_fraction:.2%}")
    print(f"Dynamic range: {report.dynamic_range:.2f}")

Replicate Merging
-----------------

.. autofunction:: maldiamrkit.preprocessing.merge_replicates

.. autofunction:: maldiamrkit.preprocessing.detect_outlier_replicates
