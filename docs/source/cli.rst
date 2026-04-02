CLI Reference
=============

MaldiAMRKit ships a command-line interface built on `Typer <https://typer.tiangolo.com/>`_.
Three subcommands cover the most common batch-processing workflows:
preprocessing spectra into feature matrices, generating quality reports, and
building standardised dataset directories.

Command Reference
-----------------

.. click:: maldiamrkit.cli:typer_click_object
   :prog: maldiamrkit
   :nested: full

Usage Examples
--------------

Preprocess
^^^^^^^^^^

Build a CSV feature matrix from a directory of ``.txt`` spectra:

.. code-block:: bash

   maldiamrkit preprocess -i data/ -o features.csv -b 3

Save individual preprocessed spectra alongside the feature matrix:

.. code-block:: bash

   maldiamrkit preprocess -i data/ -o features.csv --save-spectra-dir processed/

Use a custom preprocessing pipeline defined in YAML:

.. code-block:: bash

   maldiamrkit preprocess -i data/ -o features.csv -p config.yaml

Quality
^^^^^^^

Generate a per-spectrum quality report (SNR, TIC, peak count, etc.):

.. code-block:: bash

   maldiamrkit quality -i data/ -o quality_report.csv

Build
^^^^^

Build a DRIAMS-like dataset directory from flat ``.txt`` spectra (default layout):

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/

Build from a Bruker binary tree:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ -l bruker

Customise Bruker-specific column names:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ -l bruker \
     --path-column SpectrumPath --target-position-column Position

Organise output into year-based subfolders:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ --year-column acquisition_date

Add extra processing handlers from a YAML config:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ --extra-handlers handlers.yaml

Pipeline Configuration
----------------------

The ``--pipeline`` option accepts a JSON or YAML file describing the
preprocessing steps. When omitted, the default pipeline is used
(see :meth:`~maldiamrkit.preprocessing.preprocessing_pipeline.PreprocessingPipeline.default`).

.. code-block:: yaml

   steps:
     - name: clip
     - name: sqrt
     - name: savgol
       params: {window_length: 11, polyorder: 3}
     - name: snip
       params: {n_iters: 20}
     - name: trim
       params: {mz_min: 2000, mz_max: 20000}
     - name: tic

See the :doc:`Quickstart Guide <quickstart>` for a full walkthrough of
building and customising preprocessing pipelines.
