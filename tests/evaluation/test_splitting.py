"""Tests for stratified splitting utilities."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.evaluation import (
    CaseGroupedKFold,
    SpeciesDrugStratifiedKFold,
    case_based_split,
    stratified_species_drug_split,
)


@pytest.fixture
def amr_data():
    """Synthetic AMR dataset with species and resistance labels."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame(rng.randn(n, 10), columns=[f"bin_{i}" for i in range(10)])
    species = np.array(["E. coli"] * 40 + ["K. pneumoniae"] * 40 + ["S. aureus"] * 20)
    y = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    case_ids = np.array([f"patient_{i // 3}" for i in range(n)])
    return X, y, species, case_ids


class TestStratifiedSpeciesDrugSplit:
    """Tests for species-drug stratified splitting."""

    def test_split_sizes(self, amr_data):
        X, y, species, _ = amr_data
        X_train, X_test, y_train, y_test = stratified_species_drug_split(
            X, y, species, test_size=0.2, random_state=42
        )
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert abs(len(X_test) - 20) <= 3  # ~20% of 100

    def test_no_index_overlap(self, amr_data):
        X, y, species, _ = amr_data
        X_train, X_test, _, _ = stratified_species_drug_split(
            X, y, species, test_size=0.2, random_state=42
        )
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_reproducibility(self, amr_data):
        X, y, species, _ = amr_data
        split1 = stratified_species_drug_split(X, y, species, random_state=42)
        split2 = stratified_species_drug_split(X, y, species, random_state=42)
        np.testing.assert_array_equal(split1[2], split2[2])

    def test_numpy_input(self, amr_data):
        X, y, species, _ = amr_data
        X_np = X.values
        X_train, X_test, y_train, y_test = stratified_species_drug_split(
            X_np, y, species, test_size=0.3, random_state=42
        )
        assert len(X_train) + len(X_test) == len(X_np)

    def test_species_distribution_preserved(self, amr_data):
        """Test that species distribution is roughly preserved in splits."""
        X, y, species, _ = amr_data
        X_train, X_test, y_train, y_test = stratified_species_drug_split(
            X, y, species, test_size=0.2, random_state=42
        )
        # Both splits should contain all species
        train_species = species[X_train.index]
        test_species = species[X_test.index]
        assert set(train_species) == set(test_species)


class TestCaseBasedSplit:
    """Tests for case/patient-based splitting."""

    def test_split_sizes(self, amr_data):
        X, y, _, case_ids = amr_data
        X_train, X_test, y_train, y_test = case_based_split(
            X, y, case_ids, test_size=0.2, random_state=42
        )
        assert len(X_train) + len(X_test) == len(X)

    def test_no_case_leakage(self, amr_data):
        X, y, _, case_ids = amr_data
        X_train, X_test, _, _ = case_based_split(
            X, y, case_ids, test_size=0.3, random_state=42
        )
        train_cases = set(case_ids[X_train.index])
        test_cases = set(case_ids[X_test.index])
        assert len(train_cases & test_cases) == 0

    def test_reproducibility(self, amr_data):
        X, y, _, case_ids = amr_data
        split1 = case_based_split(X, y, case_ids, random_state=42)
        split2 = case_based_split(X, y, case_ids, random_state=42)
        np.testing.assert_array_equal(split1[2], split2[2])


class TestSpeciesDrugStratifiedKFold:
    """Tests for species-drug stratified K-fold CV."""

    def test_n_splits(self, amr_data):
        X, y, species, _ = amr_data
        cv = SpeciesDrugStratifiedKFold(n_splits=3, random_state=42)
        assert cv.get_n_splits() == 3
        folds = list(cv.split(X, y, species=species))
        assert len(folds) == 3

    def test_no_overlap(self, amr_data):
        X, y, species, _ = amr_data
        cv = SpeciesDrugStratifiedKFold(n_splits=3, random_state=42)
        for train_idx, test_idx in cv.split(X, y, species=species):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_full_coverage(self, amr_data):
        X, y, species, _ = amr_data
        cv = SpeciesDrugStratifiedKFold(n_splits=3, random_state=42)
        all_test = set()
        for _, test_idx in cv.split(X, y, species=species):
            all_test.update(test_idx)
        assert all_test == set(range(len(X)))

    def test_without_species(self, amr_data):
        """Falls back to plain stratified KFold when species=None."""
        X, y, _, _ = amr_data
        cv = SpeciesDrugStratifiedKFold(n_splits=3, random_state=42)
        folds = list(cv.split(X, y, species=None))
        assert len(folds) == 3


class TestCaseGroupedKFold:
    """Tests for case-grouped K-fold CV."""

    def test_n_splits(self, amr_data):
        X, y, _, case_ids = amr_data
        cv = CaseGroupedKFold(n_splits=5)
        folds = list(cv.split(X, y, groups=case_ids))
        assert len(folds) == 5

    def test_no_case_leakage(self, amr_data):
        X, y, _, case_ids = amr_data
        cv = CaseGroupedKFold(n_splits=5)
        for train_idx, test_idx in cv.split(X, y, groups=case_ids):
            train_cases = set(case_ids[train_idx])
            test_cases = set(case_ids[test_idx])
            assert len(train_cases & test_cases) == 0

    def test_requires_groups(self, amr_data):
        X, y, _, _ = amr_data
        cv = CaseGroupedKFold(n_splits=3)
        with pytest.raises(ValueError, match="groups"):
            list(cv.split(X, y))


class TestBuildStrataRareMerging:
    """Test rare strata merging in _build_strata."""

    def test_rare_strata_merged(self):
        from maldiamrkit.evaluation.splitting import _build_strata

        y = np.array(["R", "R", "R", "R", "S"])
        species = np.array(["A", "A", "A", "A", "B"])
        strata = _build_strata(y, species, min_count=2)
        # Rare strata are now merged per resistance label
        assert any("__rare_" in s for s in strata)


class TestBuildStrataEdgeCases:
    """Edge-case tests for _build_strata helper."""

    def test_all_same_species_single_stratum(self):
        """All same species + same label => single stratum."""
        from maldiamrkit.evaluation.splitting import _build_strata

        y = np.array(["R", "R", "R", "R"])
        species = np.array(["A", "A", "A", "A"])
        strata = _build_strata(y, species, min_count=1)
        assert len(set(strata)) == 1

    def test_min_count_1_no_merging(self):
        """min_count=1 should never merge strata."""
        from maldiamrkit.evaluation.splitting import _build_strata

        y = np.array(["R", "S", "R", "S"])
        species = np.array(["A", "A", "B", "B"])
        strata = _build_strata(y, species, min_count=1)
        # Should have 4 unique strata (A_R, A_S, B_R, B_S)
        assert len(set(strata)) == 4


class TestCaseGroupedKFoldEdgeCases:
    """Edge-case tests for CaseGroupedKFold."""

    def test_get_n_splits_returns_configured_value(self):
        """get_n_splits should return the configured n_splits."""
        cv = CaseGroupedKFold(n_splits=7)
        assert cv.get_n_splits() == 7


class TestCaseBasedSplitNumpy:
    """Test case_based_split with numpy array X."""

    def test_numpy_array_input(self, amr_data):
        X, y, _, case_ids = amr_data
        X_np = X.values if hasattr(X, "values") else np.array(X)
        from maldiamrkit.evaluation.splitting import case_based_split

        X_train, X_test, y_train, y_test = case_based_split(
            X_np, y, case_ids, test_size=0.2, random_state=42
        )
        assert len(X_train) + len(X_test) == len(X_np)
