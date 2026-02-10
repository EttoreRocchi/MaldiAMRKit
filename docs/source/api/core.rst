Core Module
===========

Core data structures for MALDI-TOF mass spectrometry analysis.

MaldiSpectrum
-------------

.. autoclass:: maldiamrkit.MaldiSpectrum
   :members:
   :undoc-members:
   :show-inheritance:

MaldiSet
--------

.. autoclass:: maldiamrkit.MaldiSet
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

Filters
-------

Composable filter system for selecting spectra from a :class:`MaldiSet`.
Filters can be combined with ``&`` (and), ``|`` (or), and ``~`` (invert).

.. autoclass:: maldiamrkit.filters.SpectrumFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.filters.SpeciesFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.filters.QualityFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.filters.DrugFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.filters.MetadataFilter
   :members:
   :undoc-members:
   :show-inheritance:

Filter Example
~~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit.filters import SpeciesFilter, DrugFilter, QualityFilter, MetadataFilter

    # Single species
    f = SpeciesFilter("Escherichia coli")

    # Multiple species with quality threshold
    f = SpeciesFilter(["E. coli", "K. pneumoniae"]) & QualityFilter(min_snr=5.0)

    # Filter by antibiotic resistance status
    f = SpeciesFilter("E. coli") & DrugFilter("Ceftriaxone", status="R")

    # Negate a filter
    f = ~SpeciesFilter("Staphylococcus aureus")

    # Custom metadata condition
    f = MetadataFilter("batch_id", lambda v: v == "batch_1")

    # Apply to a MaldiSet
    filtered_ds = ds.filter(f)
