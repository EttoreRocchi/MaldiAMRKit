CLI reference
=============

MaldiAMRKit ships a command-line interface built on `Typer <https://typer.tiangolo.com/>`_.
Three subcommands cover the most common batch-processing workflows:
preprocessing spectra into feature matrices, generating quality reports, and
building standardised dataset directories.

Command reference
-----------------

.. typer:: maldiamrkit.cli:app
   :prog: maldiamrkit
   :width: 80
   :show-nested:

Usage examples
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

A spectrum ID found in more than one input file (e.g. ``2018/x.txt`` and
``2019/x.txt``) is not built: it counts as failed in the build summary, and
the log names its files.

Build from a Bruker binary tree. The metadata needs the columns
``Identifier``, ``Path``, ``target_position`` and ``Year``, which can be
renamed with ``--id-column``, ``--path-column``,
``--target-position-column`` and ``--year-column``:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ -l bruker

Paths may start with the name of the ``-s`` folder (MARISMa writes
``/MARISMa/2024/...``). Metadata rows that repeat one spectrum (same
identifier, target position and path) are merged into one row: susceptibility
calls on which they disagree become missing, and other differing values are
joined with ``"; "``. ``--duplicate-strategy`` (``first``, ``last``, ``drop``,
``keep_all`` or ``average``) then handles the spectra of one identifier, and
a Bruker folder named by several identifiers is not built. The processed
``pdata/1/1r`` binary is read by default; ``--bruker-source fid`` reads the
raw ``fid``:

.. code-block:: bash

   maldiamrkit build -s MARISMa/ -m MARISMa/AMR.csv -o output/ -l bruker \
     --duplicate-strategy keep_all --bruker-source fid

Customise Bruker-specific column names:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ -l bruker \
     --path-column SpectrumPath --target-position-column Position

Organise output into year-based subfolders:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ --year-column acquisition_date

Add extra processing handlers from a YAML or JSON config:

.. code-block:: bash

   maldiamrkit build -s data/ -m meta.csv -o output/ --extra-handlers handlers.yaml

The config is a list of handlers. Each one names an output folder and its
``kind`` (``preprocessed`` or ``binned``), with an optional ``pipeline`` file,
read relative to the config, and a ``bin_width`` for binned output:

.. code-block:: yaml

   - folder_name: binned_3000
     kind: binned
     bin_width: 6
   - folder_name: preprocessed_sqrt
     kind: preprocessed
     pipeline: sqrt_pipeline.yaml

Pipeline configuration
----------------------

The ``--pipeline`` option accepts a JSON or YAML file describing the
preprocessing steps. When omitted, the default pipeline is used
(see :meth:`~maldiamrkit.preprocessing.PreprocessingPipeline.default`).

.. code-block:: yaml

   steps:
     - name: ClipNegatives
       step_name: clip
     - name: SqrtTransform
       step_name: sqrt
     - name: SavitzkyGolaySmooth
       step_name: smooth
       window_length: 11
       polyorder: 3
     - name: SNIPBaseline
       step_name: baseline
       half_window: 40
     - name: MzTrimmer
       step_name: trim
       mz_min: 2000
       mz_max: 20000
     - name: TICNormalizer
       step_name: normalize

``name`` is a registered transformer class, ``step_name`` a label of your
choice, and the other keys are constructor arguments.
:meth:`~maldiamrkit.preprocessing.PreprocessingPipeline.to_yaml` writes this
format from an existing pipeline.

See the :doc:`Quickstart Guide <quickstart>` for a full walkthrough of
building and customising preprocessing pipelines.
