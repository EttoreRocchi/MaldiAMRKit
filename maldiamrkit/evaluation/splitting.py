"""Stratified splitting utilities for AMR datasets.

Provides species-drug stratified and case-based (patient-grouped) splitting
to prevent data leakage and ensure balanced evaluation of AMR classifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


def _build_strata(
    y: np.ndarray,
    species: np.ndarray,
    min_count: int = 2,
) -> np.ndarray:
    """Build stratification labels from species + drug resistance.

    Combines species and resistance label into a single stratum key.
    Rare strata (< min_count samples) are merged into an "other" group.

    Parameters
    ----------
    y : array-like
        Resistance labels.
    species : array-like
        Species labels.
    min_count : int, default=2
        Minimum samples per stratum. Smaller groups are merged.

    Returns
    -------
    np.ndarray
        Array of stratum keys (strings).
    """
    y = np.asarray(y, dtype=str)
    species = np.asarray(species, dtype=str)
    strata = np.array([f"{s}__{lab}" for s, lab in zip(species, y, strict=True)])

    # Merge rare strata
    unique, counts = np.unique(strata, return_counts=True)
    rare = set(unique[counts < min_count])
    if rare:
        strata = np.array([s if s not in rare else "__rare__" for s in strata])

    return strata


def stratified_species_drug_split(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    species: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
    min_count: int = 2,
) -> tuple:
    """Stratified train/test split preserving species-drug label distributions.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like
        Resistance labels.
    species : array-like
        Species labels aligned with X.
    test_size : float, default=0.2
        Fraction of samples for the test set.
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_count : int, default=2
        Minimum samples per species-drug stratum. Smaller groups are merged.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split data.
    """
    y = np.asarray(y)
    species = np.asarray(species)
    strata = _build_strata(y, species, min_count)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, strata))

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]

    return X_train, X_test, y[train_idx], y[test_idx]


def case_based_split(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple:
    """Train/test split keeping all samples from the same patient together.

    Prevents data leakage from having the same patient in both train and test.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like
        Resistance labels.
    case_ids : array-like
        Patient/case identifiers aligned with X.
    test_size : float, default=0.2
        Fraction of groups for the test set.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split data.
    """
    y = np.asarray(y)
    case_ids = np.asarray(case_ids)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=case_ids))

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]

    return X_train, X_test, y[train_idx], y[test_idx]


class SpeciesDrugStratifiedKFold:
    """K-fold cross-validation with species-drug stratification.

    Ensures each fold preserves the distribution of species-drug combinations.
    Implements the sklearn splitter interface.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_count : int, default=2
        Minimum samples per stratum before merging.

    Examples
    --------
    >>> cv = SpeciesDrugStratifiedKFold(n_splits=5)
    >>> for train_idx, test_idx in cv.split(X, y, species=species):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
        min_count: int = 2,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_count = min_count

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits

    def split(self, X, y, species=None, groups=None):
        """Generate train/test indices for each fold.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Resistance labels.
        species : array-like
            Species labels. If None, falls back to plain stratified KFold.
        groups : ignored
            Not used, present for API compatibility.

        Yields
        ------
        train_idx, test_idx : np.ndarray
            Indices for train and test sets.
        """
        y = np.asarray(y)

        if species is not None:
            species = np.asarray(species)
            strata = _build_strata(y, species, self.min_count)
        else:
            strata = y

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        yield from skf.split(X, strata)


class CaseGroupedKFold:
    """K-fold cross-validation keeping patient cases together.

    All samples from the same case/patient are always in the same fold.
    Implements the sklearn splitter interface.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.

    Examples
    --------
    >>> cv = CaseGroupedKFold(n_splits=5)
    >>> for train_idx, test_idx in cv.split(X, y, groups=case_ids):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for each fold.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like, optional
            Labels (passed through but not used for splitting).
        groups : array-like
            Case/patient identifiers. Required.

        Yields
        ------
        train_idx, test_idx : np.ndarray
            Indices for train and test sets.

        Raises
        ------
        ValueError
            If groups is None.
        """
        if groups is None:
            raise ValueError("groups (case_ids) must be provided for CaseGroupedKFold")

        gkf = GroupKFold(n_splits=self.n_splits)
        yield from gkf.split(X, y, groups=groups)
