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

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
   cd MaldiAMRKit
   pip install -e .[dev]
