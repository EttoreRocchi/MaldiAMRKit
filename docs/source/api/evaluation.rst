Evaluation module
=================

.. py:module:: maldiamrkit.evaluation

AMR-specific evaluation metrics and stratified splitting utilities, following
EUCAST conventions.

Metrics
-------

.. autofunction:: maldiamrkit.evaluation.very_major_error_rate

.. autofunction:: maldiamrkit.evaluation.major_error_rate

.. autofunction:: maldiamrkit.evaluation.sensitivity_score

.. autofunction:: maldiamrkit.evaluation.specificity_score

.. autofunction:: maldiamrkit.evaluation.categorical_agreement

.. autofunction:: maldiamrkit.evaluation.vme_me_curve

.. autofunction:: maldiamrkit.evaluation.amr_classification_report

.. autofunction:: maldiamrkit.evaluation.amr_multilabel_report

.. autofunction:: maldiamrkit.evaluation.mic_regression_report

Sklearn scorers
~~~~~~~~~~~~~~~

Pre-built scorers for use with ``cross_val_score`` or ``GridSearchCV``:

.. py:data:: maldiamrkit.evaluation.vme_scorer

   Scorer that minimises VME (Very Major Error rate). ``y`` must hold binary
   labels (R=1, S=0), and scores are negated, so higher is better.
   Use with ``cross_val_score(pipe, X, y, scoring=vme_scorer)``.

.. py:data:: maldiamrkit.evaluation.me_scorer

   Scorer that minimises ME (Major Error rate). ``y`` must hold binary
   labels (R=1, S=0), and scores are negated, so higher is better.
   Use with ``cross_val_score(pipe, X, y, scoring=me_scorer)``.

Metrics example
~~~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit.evaluation import (
        very_major_error_rate, major_error_rate,
        amr_classification_report, vme_scorer,
    )
    from sklearn.model_selection import cross_val_score

    # y_true, y_pred, y: binary labels (R=1, S=0), for example from LabelEncoder
    # Individual metrics
    vme = very_major_error_rate(y_true, y_pred)
    me = major_error_rate(y_true, y_pred)

    # Full report
    report = amr_classification_report(y_true, y_pred)

    # Use scorer in cross-validation
    # (with replicates, use SpeciesDrugStratifiedKFold and groups=data.groups)
    scores = cross_val_score(pipe, X, y, cv=5, scoring=vme_scorer)

Splitting utilities
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

Splitting example
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from maldiamrkit.evaluation import (
        stratified_species_drug_split,
        case_based_split,
        SpeciesDrugStratifiedKFold,
        CaseGroupedKFold,
    )

    # species_labels, patient_ids: the species and the patient of each row of X

    # Single split preserving species-drug distributions, with the
    # replicates of each isolate on one side
    X_train, X_test, y_train, y_test = stratified_species_drug_split(
        X, y, species=species_labels, test_size=0.2, random_state=42,
        groups=data.groups,
    )

    # Patient-grouped split
    X_train, X_test, y_train, y_test = case_based_split(
        X, y, case_ids=patient_ids, test_size=0.2
    )

    # Sklearn-compatible CV splitters
    cv = SpeciesDrugStratifiedKFold(n_splits=5)
    for train_idx, test_idx in cv.split(X, y, groups=data.groups, species=species_labels):
        pass

    cv = CaseGroupedKFold(n_splits=5)
    for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
        pass

Multi-drug evaluation
---------------------

For predicting resistance to multiple antibiotics simultaneously:

.. code-block:: python

    import pandas as pd
    from maldiamrkit.susceptibility import LabelEncoder
    from maldiamrkit.evaluation import amr_multilabel_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier

    # Encode multi-drug labels (intermediate -> susceptible by default)
    y_encoded = LabelEncoder().fit_transform(data.y)  # one column per drug
    X_train, X_test, y_train, y_test = train_test_split(data.X, y_encoded)

    # Train multi-output model
    clf = MultiOutputClassifier(RandomForestClassifier())
    clf.fit(X_train, y_train)

    # The report needs a DataFrame with the same index and columns as y_test
    y_pred = pd.DataFrame(
        clf.predict(X_test), index=y_test.index, columns=y_test.columns
    )

    # Per-drug AMR report
    report = amr_multilabel_report(y_test, y_pred, as_dataframe=True)
    print(report)
