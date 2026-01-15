Installation
============

Requirements
------------

MaldiAMRKit requires Python 3.10 or later and the following dependencies:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- pybaselines
- gudhi
- fastdtw
- joblib

Installing from PyPI
--------------------

The easiest way to install MaldiAMRKit is via pip:

.. code-block:: bash

   pip install maldiamrkit

Installing from source
----------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiAMRKit.git
   cd MaldiAMRKit
   pip install -e .

Optional: Documentation dependencies
------------------------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"

Then build with:

.. code-block:: bash

   cd docs
   make html

Optional: Development dependencies
----------------------------------

For development (testing, linting):

.. code-block:: bash

   pip install -e ".[dev]"
