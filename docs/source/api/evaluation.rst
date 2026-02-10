Evaluation Module
=================

AMR-specific evaluation metrics, stratified splitting utilities, and label
encoding following EUCAST/CLSI conventions.

Metrics
-------

.. autofunction:: maldiamrkit.evaluation.very_major_error_rate

.. autofunction:: maldiamrkit.evaluation.major_error_rate

.. autofunction:: maldiamrkit.evaluation.sensitivity_score

.. autofunction:: maldiamrkit.evaluation.specificity_score

.. autofunction:: maldiamrkit.evaluation.categorical_agreement

.. autofunction:: maldiamrkit.evaluation.vme_me_curve

.. autofunction:: maldiamrkit.evaluation.amr_classification_report

Sklearn Scorers
~~~~~~~~~~~~~~~

Pre-built scorers for use with ``cross_val_score`` or ``GridSearchCV``:

.. py:data:: maldiamrkit.evaluation.vme_scorer

   Scorer that minimizes VME (Very Major Error rate).
   Use with ``cross_val_score(pipe, X, y, scoring=vme_scorer)``.

.. py:data:: maldiamrkit.evaluation.me_scorer

   Scorer that minimizes ME (Major Error rate).
   Use with ``cross_val_score(pipe, X, y, scoring=me_scorer)``.

Metrics Example
~~~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit.evaluation import (
        very_major_error_rate, major_error_rate,
        amr_classification_report, vme_scorer,
    )
    from sklearn.model_selection import cross_val_score

    # Individual metrics
    vme = very_major_error_rate(y_true, y_pred)
    me = major_error_rate(y_true, y_pred)

    # Full report
    report = amr_classification_report(y_true, y_pred)

    # Use scorer in cross-validation
    scores = cross_val_score(pipe, X, y, cv=5, scoring=vme_scorer)

Splitting Utilities
-------------------

.. autofunction:: maldiamrkit.evaluation.stratified_species_drug_split

.. autofunction:: maldiamrkit.evaluation.case_based_split

.. autoclass:: maldiamrkit.evaluation.SpeciesDrugStratifiedKFold
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maldiamrkit.evaluation.CaseGroupedKFold
   :members:
   :undoc-members:
   :show-inheritance:

Splitting Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit.evaluation import (
        stratified_species_drug_split,
        case_based_split,
        SpeciesDrugStratifiedKFold,
        CaseGroupedKFold,
    )

    # Single split preserving species-drug distributions
    X_train, X_test, y_train, y_test = stratified_species_drug_split(
        X, y, species=species_labels, test_size=0.2, random_state=42
    )

    # Patient-grouped split
    X_train, X_test, y_train, y_test = case_based_split(
        X, y, case_ids=patient_ids, test_size=0.2
    )

    # Sklearn-compatible CV splitters
    cv = SpeciesDrugStratifiedKFold(n_splits=5)
    for train_idx, test_idx in cv.split(X, y, species=species_labels):
        pass

    cv = CaseGroupedKFold(n_splits=5)
    for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
        pass

Label Encoding
--------------

.. autoclass:: maldiamrkit.evaluation.LabelEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Label Encoding Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit.evaluation import LabelEncoder

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
