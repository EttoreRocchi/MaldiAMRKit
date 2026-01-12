Installation
============

Requirements
------------

MaldiAMRKit requires Python 3.9 or later and the following dependencies:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- pybaselines
- gudhi
- fastdtw

Installing from source
----------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/yourusername/MaldiAMRKit.git
   cd MaldiAMRKit
   pip install -e .

Installing dependencies
-----------------------

Install all required dependencies:

.. code-block:: bash

   pip install numpy pandas scipy scikit-learn matplotlib seaborn pybaselines gudhi fastdtw

Optional: Documentation dependencies
------------------------------------

To build the documentation locally:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints nbsphinx

Then build with:

.. code-block:: bash

   cd docs
   make html
