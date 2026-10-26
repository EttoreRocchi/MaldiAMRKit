"""Stratified splitting utilities for AMR datasets.

Provides species-drug stratified and case-based (patient-grouped) splitting
for balanced evaluation of AMR classifiers. Stratification balances the
species-drug strata; grouping (``groups=``, for example the isolate of each
replicate) keeps related samples on one side of the split, which is what
prevents leakage between replicates.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


def _by_row(values, X, name: str) -> np.ndarray:
    """Return ``values`` as an array in the row order of ``X``.

    A pandas Series indexed by the row labels of a DataFrame ``X`` is
    matched to its rows by label. One with the default ``RangeIndex``, with
    an index of another depth than ``X.index``, or sharing no label with
    it, is taken by position, like an array.

    Raises
    ------
    ValueError
        If the Series index covers only part of the rows of ``X``, or
        repeats labels.
    """
    if isinstance(values, pd.Series) and isinstance(X, pd.DataFrame):
        if values.index.equals(X.index):
            return values.to_numpy()
        positional = values.index.equals(pd.RangeIndex(len(values)))
        if positional or values.index.nlevels != X.index.nlevels:
            return values.to_numpy()
        shared = X.index.isin(values.index)
        if shared.all() and values.index.is_unique:
            return values.loc[X.index].to_numpy()
        if shared.any():
            raise ValueError(
                f"{name} is a Series whose index does not match the rows of X: "
                "give one value per row of X, indexed like X."
            )
    return np.asarray(values)


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

    # Merge rare strata, preserving resistance label to maintain
    # class balance across the merged group.
    unique, counts = np.unique(strata, return_counts=True)
    rare = set(unique[counts < min_count])
    if rare:
        strata = np.array(
            [
                s if s not in rare else f"__rare_{lab}__"
                for s, lab in zip(strata, y, strict=True)
            ]
        )

    return strata


def stratified_species_drug_split(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    species: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
    min_count: int = 2,
    *,
    groups: np.ndarray | None = None,
) -> tuple:
    """Stratified train/test split preserving species-drug label distributions.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like
        Resistance labels.
    species : array-like
        Species labels aligned with X (a Series indexed like X is matched
        to its rows by label).
    test_size : float, default=0.2
        Fraction of samples for the test set.
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_count : int, default=2
        Minimum samples per species-drug stratum. Smaller groups are merged.
    groups : array-like or None, default=None
        Group of each sample (for example ``MaldiSet.groups``, the isolate
        of each replicate). When given, every group lands on one side of
        the split: the test set is the first fold of a stratified group
        k-fold with ``round(1 / test_size)`` folds, so its share is about
        ``1 / round(1 / test_size)`` of the samples (less exactly when the
        groups differ in size). A Series indexed like X is matched to its
        rows by label.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split data.

    Raises
    ------
    ValueError
        If ``groups`` is given with a ``test_size`` that is not a fraction.

    Warns
    -----
    UserWarning
        If, with ``groups``, the test share obtained differs from
        ``test_size`` by more than 0.05.
    """
    y = np.asarray(y)
    species = _by_row(species, X, "species")
    strata = _build_strata(y, species, min_count)

    if groups is None:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, strata))
    else:
        if (
            isinstance(test_size, numbers.Integral)
            or not isinstance(test_size, numbers.Real)
            or not 0 < test_size < 1
        ):
            raise ValueError(
                "With groups, test_size must be a fraction between 0 and 1, "
                f"got {test_size!r}."
            )
        n_splits = max(2, round(1 / test_size))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        train_idx, test_idx = next(
            splitter.split(X, strata, _by_row(groups, X, "groups"))
        )
        share = len(test_idx) / len(strata)
        if abs(share - test_size) > 0.05:
            warnings.warn(
                f"With groups, the test share is {share:.2f} instead of "
                f"test_size={float(test_size)} (whole groups go to one side).",
                UserWarning,
                stacklevel=2,
            )

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
        Patient/case identifiers aligned with X (a Series indexed like X is
        matched to its rows by label).
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
    case_ids = _by_row(case_ids, X, "case_ids")

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


class SpeciesDrugStratifiedKFold(BaseCrossValidator):
    """K-fold cross-validation with species-drug stratification.

    Ensures each fold preserves the distribution of species-drug combinations.
    Implements the sklearn splitter interface. When isolates have several
    replicates, pass ``groups`` (for example ``MaldiSet.groups``) so that
    each isolate stays in one fold.

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
    >>> for train_idx, test_idx in cv.split(X, y, groups=groups, species=species):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """

    # Under scikit-learn metadata routing, ``groups`` reaches ``split``.
    __metadata_request__split = {"groups": True}

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

    def get_n_splits(
        self,
        X: pd.DataFrame | np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        *,
        species: np.ndarray | None = None,
    ) -> int:
        """Return the number of splits.

        Parameters
        ----------
        X, y, groups, species : ignored
            Present for sklearn API compatibility (a search passes the
            routed split parameters here too).

        Returns
        -------
        int
            The number of folds, ``n_splits``.
        """
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        *,
        species: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each fold.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Resistance labels. Required.
        groups : array-like or None, default=None
            Group of each sample (for example ``MaldiSet.groups``, the
            isolate of each replicate). When given, every group stays in
            one fold (stratified group k-fold on the species-drug strata).
            A Series indexed like X is matched to its rows by label. With
            scikit-learn metadata routing enabled, ``groups`` given to
            ``cross_validate`` or a search reaches this method.
        species : array-like or None, default=None
            Species labels (keyword only; a Series indexed like X is matched
            to its rows by label). If None, folds are stratified on ``y``
            only.

        Yields
        ------
        train_idx, test_idx : np.ndarray
            Indices for train and test sets.

        Raises
        ------
        ValueError
            If ``y`` is None.
        """
        if y is None:
            raise ValueError(
                "y must be provided for SpeciesDrugStratifiedKFold stratification"
            )
        y = np.asarray(y)

        if species is not None:
            species = _by_row(species, X, "species")
            strata = _build_strata(y, species, self.min_count)
        else:
            strata = y

        if groups is not None:
            sgkf = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None,
            )
            yield from sgkf.split(X, strata, _by_row(groups, X, "groups"))
            return

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        yield from skf.split(X, strata)


class CaseGroupedKFold(BaseCrossValidator):
    """K-fold cross-validation keeping patient cases together and stratified by ``y``.

    All samples from the same case/patient are always in the same fold,
    and folds are stratified on the resistance label to preserve class
    balance. Wraps :class:`sklearn.model_selection.StratifiedGroupKFold`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle group order before splitting.
    random_state : int or None, default=None
        Random seed (used only when ``shuffle=True``).

    Examples
    --------
    >>> cv = CaseGroupedKFold(n_splits=5)
    >>> for train_idx, test_idx in cv.split(X, y, groups=case_ids):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """

    # Under scikit-learn metadata routing, ``groups`` reaches ``split``.
    __metadata_request__split = {"groups": True}

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self,
        X: pd.DataFrame | np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return the number of splits.

        Parameters
        ----------
        X, y, groups : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        int
            The number of folds, ``n_splits``.
        """
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate stratified, group-preserving train/test indices for each fold.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Resistance labels. Required for stratification.
        groups : array-like
            Case/patient identifiers. Required. A Series indexed like X is
            matched to its rows by label.

        Yields
        ------
        train_idx, test_idx : np.ndarray
            Indices for train and test sets.

        Raises
        ------
        ValueError
            If ``groups`` or ``y`` is None.
        """
        if groups is None:
            raise ValueError("groups (case_ids) must be provided for CaseGroupedKFold")
        if y is None:
            raise ValueError("y must be provided for CaseGroupedKFold stratification")

        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        yield from sgkf.split(X, y, groups=_by_row(groups, X, "groups"))
