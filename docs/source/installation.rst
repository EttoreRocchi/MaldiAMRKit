Installation
============

.. code-block:: bash

   pip install maldiamrkit

Optional: Batch Correction & UMAP
---------------------------------

For multi-site batch effect correction with
`combatlearn <https://combatlearn.readthedocs.io/>`__ and
UMAP-based exploratory plots:

.. code-block:: bash

   pip install maldiamrkit[batch]

This installs `combatlearn <https://github.com/EttoreRocchi/combatlearn>`__
for ComBat-based batch correction and ``umap-learn`` for UMAP
dimensionality reduction.

See the `combatlearn documentation <https://combatlearn.readthedocs.io/>`__
for usage examples. Reference:

   Rocchi, E., Nicitra, E., Calvo, M. et al. *Combining mass spectrometry
   and machine learning models for predicting Klebsiella pneumoniae
   antimicrobial resistance: a multicenter experience from clinical
   isolates in Italy*. **BMC Microbiol** (2026).
   `doi:10.1186/s12866-025-04657-2
   <https://link.springer.com/article/10.1186/s12866-025-04657-2>`_

Optional: extended mzML/mzXML support
-------------------------------------

``.mzML`` and ``.mzXML`` files (optionally gzipped) are read out of the box
by :func:`maldiamrkit.io.read_spectrum` that handles zlib-compressed
or uncompressed 32/64-bit arrays. For numpress-compressed arrays install the
``formats`` extra, which adds `pyteomics <https://pyteomics.readthedocs.io/>`__
with ``lxml``, ``psims``, and ``pynumpress``:

.. code-block:: bash

   pip install maldiamrkit[formats]

``pynumpress`` ships no wheels for Python 3.10, so numpress arrays need
Python 3.11 or newer.

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
   cd MaldiAMRKit
   pip install -e .[dev]
