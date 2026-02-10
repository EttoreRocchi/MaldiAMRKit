"""Unit tests for composable filter system."""

import numpy as np
import pandas as pd
import pytest

from maldiamrkit.filters import (
    DrugFilter,
    MetadataFilter,
    QualityFilter,
    SpeciesFilter,
    SpectrumFilter,
)


@pytest.fixture
def sample_meta() -> pd.DataFrame:
    """Metadata DataFrame for testing filters."""
    return pd.DataFrame(
        {
            "ID": ["s1", "s2", "s3", "s4", "s5"],
            "Species": [
                "Escherichia coli",
                "Klebsiella pneumoniae",
                "Escherichia coli",
                "Staphylococcus aureus",
                "Klebsiella pneumoniae",
            ],
            "snr": [10.0, 3.0, 8.0, 1.5, 6.0],
            "n_peaks": [50, 20, 45, 10, 35],
            "baseline_fraction": [0.1, 0.5, 0.2, 0.8, 0.3],
            "batch_id": ["b1", "b1", "b2", "b2", "b1"],
            "Ceftriaxone": ["R", "S", "R", np.nan, "S"],
            "Ceftazidime": ["S", np.nan, "I", "R", "S"],
        }
    ).set_index("ID")


class TestSpeciesFilter:
    """Tests for SpeciesFilter."""

    def test_single_species(self, sample_meta: pd.DataFrame):
        f = SpeciesFilter("Escherichia coli")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3"]

    def test_multiple_species(self, sample_meta: pd.DataFrame):
        f = SpeciesFilter(["Escherichia coli", "Klebsiella pneumoniae"])
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s2", "s3", "s5"]

    def test_no_match(self, sample_meta: pd.DataFrame):
        f = SpeciesFilter("Pseudomonas aeruginosa")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == []

    def test_custom_column(self):
        meta = pd.DataFrame({"ID": ["a"], "organism": ["E. coli"]}).set_index("ID")
        f = SpeciesFilter("E. coli", column="organism")
        assert f(meta.iloc[0])

    def test_repr(self):
        assert "Escherichia coli" in repr(SpeciesFilter("Escherichia coli"))


class TestQualityFilter:
    """Tests for QualityFilter."""

    def test_min_snr(self, sample_meta: pd.DataFrame):
        f = QualityFilter(min_snr=5.0)
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3", "s5"]

    def test_min_peaks(self, sample_meta: pd.DataFrame):
        f = QualityFilter(min_peaks=30)
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3", "s5"]

    def test_max_baseline_fraction(self, sample_meta: pd.DataFrame):
        f = QualityFilter(max_baseline_fraction=0.3)
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3", "s5"]

    def test_combined_thresholds(self, sample_meta: pd.DataFrame):
        f = QualityFilter(min_snr=5.0, min_peaks=40)
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3"]

    def test_no_thresholds_keeps_all(self, sample_meta: pd.DataFrame):
        f = QualityFilter()
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert len(kept) == 5

    def test_missing_column_rejects(self):
        meta = pd.DataFrame({"ID": ["a"]}).set_index("ID")
        f = QualityFilter(min_snr=5.0)
        assert not f(meta.iloc[0])

    def test_repr(self):
        f = QualityFilter(min_snr=5.0, min_peaks=10)
        r = repr(f)
        assert "min_snr=5.0" in r
        assert "min_peaks=10" in r


class TestMetadataFilter:
    """Tests for MetadataFilter."""

    def test_equality_condition(self, sample_meta: pd.DataFrame):
        f = MetadataFilter("batch_id", lambda v: v == "b1")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s2", "s5"]

    def test_comparison_condition(self, sample_meta: pd.DataFrame):
        f = MetadataFilter("snr", lambda v: v > 7.0)
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3"]

    def test_missing_column_rejects(self):
        meta = pd.DataFrame({"ID": ["a"]}).set_index("ID")
        f = MetadataFilter("nonexistent", lambda v: True)
        assert not f(meta.iloc[0])

    def test_repr(self):
        f = MetadataFilter("batch_id", lambda v: v == "b1")
        assert "batch_id" in repr(f)

    def test_bad_condition_raises(self):
        meta = pd.DataFrame({"ID": ["a"], "batch_id": ["b1"]}).set_index("ID")
        f = MetadataFilter("batch_id", lambda v: v["nonexistent"])
        with pytest.raises(ValueError, match="condition failed"):
            f(meta.iloc[0])


class TestDrugFilter:
    """Tests for DrugFilter."""

    def test_has_data(self, sample_meta: pd.DataFrame):
        """Keep samples with non-null values for the drug."""
        f = DrugFilter("Ceftriaxone")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s2", "s3", "s5"]

    def test_single_status(self, sample_meta: pd.DataFrame):
        f = DrugFilter("Ceftriaxone", status="R")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3"]

    def test_multiple_statuses(self, sample_meta: pd.DataFrame):
        f = DrugFilter("Ceftazidime", status=["R", "I"])
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s3", "s4"]

    def test_nan_excluded(self, sample_meta: pd.DataFrame):
        """NaN values should be excluded even without status filter."""
        f = DrugFilter("Ceftazidime")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert "s2" not in kept  # s2 has NaN for Ceftazidime

    def test_missing_column(self):
        meta = pd.DataFrame({"ID": ["a"]}).set_index("ID")
        f = DrugFilter("Nonexistent")
        assert not f(meta.iloc[0])

    def test_composition_with_species(self, sample_meta: pd.DataFrame):
        f = SpeciesFilter("Escherichia coli") & DrugFilter("Ceftriaxone", status="R")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3"]

    def test_repr_no_status(self):
        assert repr(DrugFilter("Ceftriaxone")) == "DrugFilter('Ceftriaxone')"

    def test_repr_single_status(self):
        assert (
            repr(DrugFilter("Ceftriaxone", status="R"))
            == "DrugFilter('Ceftriaxone', status='R')"
        )

    def test_repr_multiple_statuses(self):
        r = repr(DrugFilter("Ceftriaxone", status=["S", "R"]))
        assert "DrugFilter('Ceftriaxone', status=" in r


class TestFilterComposition:
    """Tests for combining filters with &, |, ~."""

    def test_and(self, sample_meta: pd.DataFrame):
        f = SpeciesFilter("Escherichia coli") & QualityFilter(min_snr=9.0)
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1"]

    def test_or(self, sample_meta: pd.DataFrame):
        f = SpeciesFilter("Escherichia coli") | SpeciesFilter("Staphylococcus aureus")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3", "s4"]

    def test_not(self, sample_meta: pd.DataFrame):
        f = ~SpeciesFilter("Escherichia coli")
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s2", "s4", "s5"]

    def test_complex_composition(self, sample_meta: pd.DataFrame):
        f = (
            SpeciesFilter("Escherichia coli") | SpeciesFilter("Klebsiella pneumoniae")
        ) & QualityFilter(min_snr=5.0)
        kept = [sid for sid, row in sample_meta.iterrows() if f(row)]
        assert kept == ["s1", "s3", "s5"]

    def test_repr_composed(self):
        f = SpeciesFilter("E. coli") & QualityFilter(min_snr=5.0)
        r = repr(f)
        assert "&" in r

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SpectrumFilter()
