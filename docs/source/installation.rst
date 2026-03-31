Installation
============

.. code-block:: bash

   pip install maldiamrkit

Optional: mzML/mzXML Format Support
------------------------------------

To read mzML and mzXML files (common in clinical mass spectrometry),
install with the ``formats`` extra:

.. code-block:: bash

   pip install maldiamrkit[formats]

This installs `pyteomics <https://pyteomics.readthedocs.io/>`_ and
``lxml`` for parsing standard mass spectrometry data formats.

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

Optional: All Extras
--------------------

Install all optional dependencies at once:

.. code-block:: bash

   pip install maldiamrkit[all]

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
   cd MaldiAMRKit
   pip install -e .[dev]
