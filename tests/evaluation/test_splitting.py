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

    def test_requires_y_for_stratification(self, amr_data):
        X, _, _, case_ids = amr_data
        cv = CaseGroupedKFold(n_splits=3)
        with pytest.raises(ValueError, match="y"):
            list(cv.split(X, y=None, groups=case_ids))

    def test_class_balance_preserved(self, amr_data):
        """Each fold should have both classes present (StratifiedGroupKFold)."""
        X, y, _, case_ids = amr_data
        cv = CaseGroupedKFold(n_splits=5)
        for train_idx, test_idx in cv.split(X, y, groups=case_ids):
            train_classes = set(np.unique(y[train_idx]))
            test_classes = set(np.unique(y[test_idx]))
            assert 0 in train_classes and 1 in train_classes
            assert 0 in test_classes and 1 in test_classes


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


class TestGroupAwareStratifiedSplits:
    """Species-drug stratified splits keep every group on one side."""

    @pytest.fixture()
    def replicates(self):
        rng = np.random.default_rng(0)
        n_iso = 40
        iso = np.repeat([f"i{k}" for k in range(n_iso)], 2)
        species = np.repeat(rng.choice(["E. coli", "K. pneumoniae"], n_iso), 2)
        y = np.repeat(rng.choice([0, 1], n_iso, p=[0.6, 0.4]), 2)
        X = pd.DataFrame(
            {"f": np.arange(len(y), dtype=float)},
            index=[f"s{k}" for k in range(len(y))],
        )
        return X, y, species, iso

    def test_kfold_honours_groups_in_cross_validate(self, replicates):
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import cross_validate

        X, y, _, iso = replicates
        cv = SpeciesDrugStratifiedKFold(n_splits=5, random_state=0)
        res = cross_validate(
            DummyClassifier(), X, y, groups=iso, cv=cv, return_indices=True
        )
        for train, test in zip(
            res["indices"]["train"], res["indices"]["test"], strict=True
        ):
            assert not set(iso[train]) & set(iso[test])

    def test_kfold_split_with_species_and_groups(self, replicates):
        X, y, species, iso = replicates
        cv = SpeciesDrugStratifiedKFold(n_splits=5, random_state=0)
        for train, test in cv.split(X, y, species=species, groups=iso):
            assert not set(iso[train]) & set(iso[test])

    def test_kfold_without_groups_is_unchanged(self, replicates):
        from sklearn.model_selection import StratifiedKFold

        from maldiamrkit.evaluation.splitting import _build_strata

        X, y, species, _ = replicates
        got = list(
            SpeciesDrugStratifiedKFold(n_splits=5, random_state=0).split(
                X, y, species=species
            )
        )
        expected = list(
            StratifiedKFold(5, shuffle=True, random_state=0).split(
                X, _build_strata(y, species, 2)
            )
        )
        for (a, b), (c, d) in zip(got, expected, strict=True):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)

    def test_single_split_with_groups(self, replicates):
        X, y, species, iso = replicates
        X_train, X_test, _, _ = stratified_species_drug_split(
            X, y, species, test_size=0.2, random_state=0, groups=iso
        )
        train_iso = set(iso[X.index.get_indexer(X_train.index)])
        test_iso = set(iso[X.index.get_indexer(X_test.index)])
        assert not train_iso & test_iso
        assert abs(len(X_test) / len(X) - 0.2) <= 0.05

    def test_single_split_with_groups_warns_on_coarse_test_size(self, replicates):
        X, y, species, iso = replicates
        with pytest.warns(UserWarning, match="test share"):
            stratified_species_drug_split(
                X, y, species, test_size=0.4, random_state=0, groups=iso
            )

    def test_single_split_with_groups_rejects_integer_test_size(self, replicates):
        X, y, species, iso = replicates
        with pytest.raises(ValueError, match="test_size"):
            stratified_species_drug_split(X, y, species, test_size=10, groups=iso)

    def test_single_split_without_groups_is_unchanged(self, replicates):
        from sklearn.model_selection import StratifiedShuffleSplit

        from maldiamrkit.evaluation.splitting import _build_strata

        X, y, species, _ = replicates
        X_train, X_test, _, _ = stratified_species_drug_split(
            X, y, species, test_size=0.2, random_state=0
        )
        train, test = next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
                X, _build_strata(y, species, 2)
            )
        )
        assert list(X_train.index) == list(X.index[train])
        assert list(X_test.index) == list(X.index[test])


class TestSplitInputs:
    """Groups and species are matched to the rows of X, however they are passed."""

    @pytest.fixture()
    def replicates(self):
        rng = np.random.default_rng(0)
        n_iso = 40
        iso = np.repeat([f"i{k}" for k in range(n_iso)], 2)
        species = np.repeat(rng.choice(["E. coli", "K. pneumoniae"], n_iso), 2)
        y = np.repeat(rng.choice([0, 1], n_iso, p=[0.6, 0.4]), 2)
        X = pd.DataFrame(
            {"f": np.arange(len(y), dtype=float)},
            index=[f"s{k}" for k in range(len(y))],
        )
        return X, y, species, iso

    def test_positional_groups_are_groups(self, replicates):
        X, y, _, iso = replicates
        cv = SpeciesDrugStratifiedKFold(n_splits=5, random_state=0)
        for train, test in cv.split(X, y, iso):
            assert not set(iso[train]) & set(iso[test])

    def test_groups_series_in_another_order(self, replicates):
        X, y, species, iso = replicates
        order = np.random.default_rng(1).permutation(len(X))
        groups = pd.Series(iso, index=X.index).iloc[order]
        cv = SpeciesDrugStratifiedKFold(n_splits=5, random_state=0)
        for train, test in cv.split(X, y, groups=groups, species=species):
            assert not set(iso[train]) & set(iso[test])
        X_train, X_test, _, _ = stratified_species_drug_split(
            X, y, species, test_size=0.2, random_state=0, groups=groups
        )
        by_row = pd.Series(iso, index=X.index)
        assert not set(by_row[X_train.index]) & set(by_row[X_test.index])

    def test_species_series_in_another_order(self, replicates):
        X, y, species, iso = replicates
        order = np.random.default_rng(2).permutation(len(X))
        shuffled = pd.Series(species, index=X.index).iloc[order]
        cv = SpeciesDrugStratifiedKFold(n_splits=5, random_state=0)
        got = list(cv.split(X, y, groups=iso, species=shuffled))
        expected = list(cv.split(X, y, groups=iso, species=species))
        for (a, b), (c, d) in zip(got, expected, strict=True):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)

    def test_series_missing_rows_raises(self, replicates):
        X, y, species, iso = replicates
        groups = pd.Series(iso, index=X.index).iloc[:-4]
        cv = SpeciesDrugStratifiedKFold(n_splits=5, random_state=0)
        with pytest.raises(ValueError, match="groups"):
            next(cv.split(X, y, groups=groups))

    def test_case_ids_series_in_another_order(self, amr_data):
        X, y, _, case_ids = amr_data
        X = X.set_axis([f"s{k}" for k in range(len(X))])
        order = np.random.default_rng(3).permutation(len(X))
        shuffled = pd.Series(case_ids, index=X.index).iloc[order]
        by_row = pd.Series(case_ids, index=X.index)
        for train, test in CaseGroupedKFold(n_splits=5, random_state=0).split(
            X, y, groups=shuffled
        ):
            assert not set(case_ids[train]) & set(case_ids[test])
        X_train, X_test, _, _ = case_based_split(
            X, y, shuffled, test_size=0.2, random_state=0
        )
        assert not set(by_row[X_train.index]) & set(by_row[X_test.index])


class TestGroupsRouting:
    """With metadata routing on, groups reach the splitters."""

    @pytest.fixture()
    def replicates(self):
        rng = np.random.default_rng(0)
        iso = np.repeat([f"i{k}" for k in range(40)], 2)
        y = np.repeat(rng.choice([0, 1], 40, p=[0.6, 0.4]), 2)
        X = rng.normal(size=(len(y), 3))
        return X, y, iso

    @pytest.mark.parametrize(
        "cv",
        [
            SpeciesDrugStratifiedKFold(n_splits=4, random_state=0),
            CaseGroupedKFold(n_splits=4, random_state=0),
        ],
    )
    def test_cross_validate_routes_groups(self, replicates, cv):
        import sklearn
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import cross_validate

        X, y, iso = replicates
        with sklearn.config_context(enable_metadata_routing=True):
            res = cross_validate(
                DummyClassifier(),
                X,
                y,
                cv=cv,
                params={"groups": iso},
                return_indices=True,
            )
        for train, test in zip(
            res["indices"]["train"], res["indices"]["test"], strict=True
        ):
            assert not set(iso[train]) & set(iso[test])

    def test_grid_search_routes_groups(self, replicates):
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV

        X, y, iso = replicates
        cv = SpeciesDrugStratifiedKFold(n_splits=4, random_state=0)
        search = GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, cv=cv)
        with sklearn.config_context(enable_metadata_routing=True):
            search.fit(X, y, groups=iso)
        assert search.n_splits_ == 4


class TestActualTestShare:
    """The test-share warning compares the share obtained with test_size."""

    def test_skewed_groups_warn(self):
        rng = np.random.default_rng(0)
        iso = np.array(["big1"] * 40 + ["big2"] * 30 + [f"s{k}" for k in range(40)])
        y = rng.choice([0, 1], len(iso))
        species = np.array(["E. coli"] * len(iso))
        X = np.arange(len(iso), dtype=float)[:, None]
        with pytest.warns(UserWarning, match="test share"):
            stratified_species_drug_split(
                X, y, species, test_size=0.2, random_state=0, groups=iso
            )

    def test_numpy_float_test_size(self):
        rng = np.random.default_rng(0)
        iso = np.repeat([f"i{k}" for k in range(40)], 2)
        y = np.repeat(rng.choice([0, 1], 40), 2)
        species = np.array(["E. coli"] * len(iso))
        X = np.arange(len(iso), dtype=float)[:, None]
        X_train, X_test, _, _ = stratified_species_drug_split(
            X, y, species, test_size=np.float32(0.2), random_state=0, groups=iso
        )
        assert len(X_test) / len(iso) == pytest.approx(0.2, abs=0.05)


class TestSplitInputChecks:
    """Aligned Series pass through and y is required."""

    def test_series_indexed_like_x_matches_the_array(self):
        rng = np.random.default_rng(0)
        iso = np.repeat([f"i{k}" for k in range(20)], 2)
        y = np.repeat(rng.choice([0, 1], 20), 2)
        X = pd.DataFrame({"f": np.arange(40.0)}, index=[f"s{k}" for k in range(40)])
        cv = SpeciesDrugStratifiedKFold(n_splits=4, random_state=0)
        got = list(cv.split(X, y, groups=pd.Series(iso, index=X.index)))
        expected = list(cv.split(X, y, groups=iso))
        for (a, b), (c, d) in zip(got, expected, strict=True):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)

    def test_y_is_required(self):
        with pytest.raises(ValueError, match="y must be provided"):
            next(SpeciesDrugStratifiedKFold().split(np.zeros((10, 1))))


class TestSplitEdgeCases:
    """Routing of species, index types and test_size checks."""

    @pytest.fixture()
    def replicates(self):
        rng = np.random.default_rng(0)
        iso = np.repeat([f"i{k}" for k in range(40)], 2)
        species = np.repeat(rng.choice(["E. coli", "K. pneumoniae"], 40), 2)
        y = np.repeat(rng.choice([0, 1], 40, p=[0.6, 0.4]), 2)
        return rng.normal(size=(len(y), 3)), y, species, iso

    def test_grid_search_routes_species(self, replicates):
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV

        X, y, species, iso = replicates
        with sklearn.config_context(enable_metadata_routing=True):
            cv = SpeciesDrugStratifiedKFold(n_splits=4, random_state=0)
            cv.set_split_request(species=True)
            search = GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, cv=cv)
            search.fit(X, y, groups=iso, species=species)
        assert search.n_splits_ == 4

    def test_range_index_series_is_positional(self, replicates):
        X, y, species, iso = replicates
        order = np.random.default_rng(4).permutation(len(y))
        X = pd.DataFrame(X).iloc[order]
        y, iso = y[order], iso[order]
        cv = SpeciesDrugStratifiedKFold(n_splits=4, random_state=0)
        for train, test in cv.split(X, y, groups=pd.Series(iso)):
            assert not set(iso[train]) & set(iso[test])

    def test_multiindex_x_with_a_plain_series(self, replicates):
        X, y, species, iso = replicates
        X = pd.DataFrame(
            X,
            index=pd.MultiIndex.from_arrays([iso, np.arange(len(iso))]),
        )
        cv = CaseGroupedKFold(n_splits=4, random_state=0)
        for train, test in cv.split(X, y, groups=pd.Series(iso)):
            assert not set(iso[train]) & set(iso[test])

    @pytest.mark.parametrize(
        "test_size", [0.0, -0.2, 1.0, 1.5, float("inf"), float("nan")]
    )
    def test_test_size_must_be_a_fraction(self, replicates, test_size):
        X, y, species, iso = replicates
        with pytest.raises(ValueError, match="between 0 and 1"):
            stratified_species_drug_split(
                X, y, species, test_size=test_size, random_state=0, groups=iso
            )
