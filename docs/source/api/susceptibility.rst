Susceptibility module
=====================

.. py:module:: maldiamrkit.susceptibility

Clinical susceptibility utilities: MIC encoding, breakpoint tables, and
R/I/S label encoding. Added in v0.15. The
:class:`~maldiamrkit.susceptibility.LabelEncoder` previously lived in the
:doc:`Evaluation module <evaluation>` and was moved here to sit alongside
the new MIC tooling; the old import path has been removed.

The regression-style evaluation function
:func:`maldiamrkit.evaluation.mic_regression_report` lives in the
:doc:`Evaluation module <evaluation>` alongside the binary AMR metrics it
complements.

MIC encoding
------------

.. autoclass:: maldiamrkit.susceptibility.MICEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Breakpoints
-----------

.. autoclass:: maldiamrkit.susceptibility.BreakpointTable
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.susceptibility.BreakpointResult
   :members:
   :undoc-members:
   :show-inheritance:

Label encoding
--------------

.. autoclass:: maldiamrkit.susceptibility.LabelEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.susceptibility.IntermediateHandling
   :members:
   :undoc-members:
   :show-inheritance:

Label encoding example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit.susceptibility import LabelEncoder

    enc = LabelEncoder()  # I -> susceptible (default)
    y_binary = enc.fit_transform(["R", "S", "I", "R", "S"])
    # array([1, 0, 0, 1, 0])

    # Treat intermediate as resistant
    enc = LabelEncoder(intermediate="resistant")
    y_binary = enc.fit_transform(["R", "S", "I"])
    # array([1, 0, 1])

    # Drop intermediate samples entirely
    enc = LabelEncoder(intermediate="drop")
    y_binary = enc.fit_transform(["R", "S", "I"])
    # array([1, 0])

MIC encoding example
~~~~~~~~~~~~~~~~~~~~

End-to-end: from raw MIC strings to ``log2(MIC)`` regression targets and
S/I/R category labels, using a bundled EUCAST breakpoint table. The
regression evaluator (:func:`maldiamrkit.evaluation.mic_regression_report`)
is imported from the :doc:`Evaluation module <evaluation>`.

Species and drug must match the names in the table exactly: bundled tables
use EUCAST entries such as ``Enterobacterales`` and
``Ceftriaxone (indications other than meningitis)``, not ``Escherichia coli``
or ``Ceftriaxone``. ``bp.species()`` and ``bp.drugs()`` list them.

.. code-block:: python

    from maldiamrkit.susceptibility import BreakpointTable, MICEncoder
    from maldiamrkit.evaluation import mic_regression_report

    # Load the latest bundled EUCAST table
    bp = BreakpointTable.from_latest()
    drug = "Ceftriaxone (indications other than meningitis)"

    # meta: DataFrame with a MIC column and a Species column of EUCAST names
    enc = MICEncoder(
        breakpoints=bp,
        species_col="Species",
        drug=drug,
    )
    targets = enc.fit_transform(meta)  # log2_mic, censored, category, atu, source

    # Evaluate regression predictions (y_pred_log2: your model's predictions)
    report = mic_regression_report(
        y_true=targets["log2_mic"],
        y_pred=y_pred_log2,
        breakpoints=bp,
        species="Enterobacterales",
        drug=drug,
    )
    print(report["rmse_log2"], report["essential_agreement"])
