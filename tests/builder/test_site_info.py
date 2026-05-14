"""Tests for the dataset manifest (``site_info.json``).

Covers:
* ``SiteInfo`` round-trip (write -> read)
* Lenient version policy: warn on newer ``format_version``, raise on
  missing required fields
* ``DRIAMSLayout`` honours the manifest when present, with explicit
  kwargs winning over manifest values, and a missing manifest falling
  back to library defaults
* ``DatasetBuilder.build()`` writes a manifest at the dataset root
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from maldiamrkit.data import (
    CURRENT_FORMAT_VERSION,
    MANIFEST_FILENAME,
    BuildInfo,
    DRIAMSLayout,
    SiteInfo,
    read_site_info,
    write_site_info,
)

# ----------------------------------------------------------- SiteInfo round-trip


def _example_site_info() -> SiteInfo:
    return SiteInfo(
        id_column="code",
        metadata_dir="id",
        metadata_suffix="_clean.csv",
        spectrum_ext=".txt",
        spectra_folders=["raw", "preprocessed", "binned_6000"],
        mz_range=(2000.0, 19997.0),
        bin_width=3.0,
        build_info=BuildInfo(
            maldiamrkit_version="0.15.0",
            created_at="2026-05-14T12:34:56Z",
            source_layout="BrukerTreeLayout",
            duplicate_strategy="first",
            n_total_spectra=100,
            n_succeeded=98,
            n_failed=2,
        ),
    )


class TestSiteInfoRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        info = _example_site_info()
        path = write_site_info(tmp_path, info)
        assert path == tmp_path / MANIFEST_FILENAME
        assert path.exists()

        loaded = read_site_info(tmp_path)
        assert loaded is not None
        assert loaded.id_column == info.id_column
        assert loaded.metadata_suffix == info.metadata_suffix
        assert loaded.spectra_folders == info.spectra_folders
        assert loaded.mz_range == info.mz_range
        assert loaded.bin_width == info.bin_width
        assert loaded.format_version == CURRENT_FORMAT_VERSION
        # build_info nested round-trip
        assert loaded.build_info is not None
        assert loaded.build_info.maldiamrkit_version == "0.15.0"
        assert loaded.build_info.n_failed == 2

    def test_missing_returns_none_by_default(self, tmp_path: Path):
        assert read_site_info(tmp_path) is None

    def test_missing_raises_when_required(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_site_info(tmp_path, missing_ok=False)

    def test_write_requires_existing_dir(self, tmp_path: Path):
        bogus = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            write_site_info(bogus, _example_site_info())

    def test_json_is_human_readable(self, tmp_path: Path):
        write_site_info(tmp_path, _example_site_info())
        text = (tmp_path / MANIFEST_FILENAME).read_text()
        # format_version is first - aids `head -1` debugging.
        first_key_line = next(
            line for line in text.splitlines() if line.strip().startswith('"')
        )
        assert '"format_version"' in first_key_line

    def test_optional_build_info_can_be_omitted(self, tmp_path: Path):
        info = SiteInfo(
            id_column="code",
            metadata_dir="id",
            metadata_suffix="_clean.csv",
            spectrum_ext=".txt",
            spectra_folders=["raw"],
            mz_range=(2000.0, 20000.0),
            bin_width=3.0,
        )
        write_site_info(tmp_path, info)
        loaded = read_site_info(tmp_path)
        assert loaded is not None
        assert loaded.build_info is None


# ---------------------------------------------------- Lenient version policy


class TestVersionPolicy:
    def _write_raw(self, tmp_path: Path, raw: dict) -> Path:
        path = tmp_path / MANIFEST_FILENAME
        path.write_text(json.dumps(raw))
        return path

    def test_future_version_emits_warning_and_still_loads(self, tmp_path: Path):
        raw = _example_site_info().to_dict()
        raw["format_version"] = CURRENT_FORMAT_VERSION + 1  # pretend a v2 manifest
        raw["something_new"] = "v2-only-field"  # should be ignored
        self._write_raw(tmp_path, raw)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = read_site_info(tmp_path)

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings, "expected a UserWarning for newer format_version"
        assert "format_version" in str(user_warnings[0].message)

        # The v1-required fields are still present, so loading succeeds.
        assert loaded is not None
        assert loaded.format_version == CURRENT_FORMAT_VERSION + 1
        # Unknown top-level field is silently ignored.
        assert not hasattr(loaded, "something_new")

    def test_missing_required_field_raises(self, tmp_path: Path):
        raw = _example_site_info().to_dict()
        del raw["metadata_suffix"]
        self._write_raw(tmp_path, raw)

        with pytest.raises(ValueError, match="metadata_suffix"):
            read_site_info(tmp_path)

    def test_missing_format_version_raises(self, tmp_path: Path):
        raw = _example_site_info().to_dict()
        del raw["format_version"]
        self._write_raw(tmp_path, raw)

        with pytest.raises(ValueError, match="format_version"):
            read_site_info(tmp_path)

    def test_non_integer_format_version_raises(self, tmp_path: Path):
        raw = _example_site_info().to_dict()
        raw["format_version"] = "1.0"  # string, not int
        self._write_raw(tmp_path, raw)

        with pytest.raises(ValueError, match="non-integer format_version"):
            read_site_info(tmp_path)

    def test_malformed_mz_range_raises(self, tmp_path: Path):
        raw = _example_site_info().to_dict()
        raw["mz_range"] = [2000.0]  # length 1, not 2
        self._write_raw(tmp_path, raw)

        with pytest.raises(ValueError, match="mz_range"):
            read_site_info(tmp_path)

    def test_invalid_json_raises(self, tmp_path: Path):
        (tmp_path / MANIFEST_FILENAME).write_text("not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            read_site_info(tmp_path)

    def test_top_level_not_object_raises(self, tmp_path: Path):
        (tmp_path / MANIFEST_FILENAME).write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="JSON object"):
            read_site_info(tmp_path)


# ----------------------------------------------------- DRIAMSLayout integration


def _make_fake_dataset(
    root: Path,
    *,
    metadata_suffix: str = "_clean.csv",
    metadata_dir: str = "id",
    year: str = "2024",
):
    """Lay down a minimal on-disk dataset that DRIAMSLayout can read."""
    id_dir = root / metadata_dir / year
    id_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "code": ["sample_a", "sample_b"],
            "Species": ["Escherichia coli", "Escherichia coli"],
        }
    )
    df.to_csv(id_dir / f"{year}{metadata_suffix}", index=False)


class TestDRIAMSLayoutManifestIntegration:
    def test_kwargs_take_precedence_over_manifest(self, tmp_path: Path):
        # Build a fake dataset with a non-default suffix on disk.
        _make_fake_dataset(tmp_path, metadata_suffix="_strat.csv")
        write_site_info(
            tmp_path,
            SiteInfo(
                id_column="code",
                metadata_dir="id",
                metadata_suffix="_clean.csv",  # manifest says "clean"...
                spectrum_ext=".txt",
                spectra_folders=["raw"],
                mz_range=(2000.0, 20000.0),
                bin_width=3.0,
            ),
        )

        # ...but explicit kwarg overrides the manifest.
        layout = DRIAMSLayout(tmp_path, metadata_suffix="_strat.csv", year="2024")
        assert layout.metadata_suffix == "_strat.csv"
        meta = layout.discover_metadata()
        assert "ID" in meta.columns
        assert len(meta) == 2

    def test_manifest_fills_unspecified_kwargs(self, tmp_path: Path):
        # The on-disk suffix is "_strat.csv"; the manifest declares it.
        _make_fake_dataset(tmp_path, metadata_suffix="_strat.csv")
        write_site_info(
            tmp_path,
            SiteInfo(
                id_column="code",
                metadata_dir="id",
                metadata_suffix="_strat.csv",
                spectrum_ext=".txt",
                spectra_folders=["raw"],
                mz_range=(2500.0, 15000.0),
                bin_width=4.0,
            ),
        )

        # User passes nothing for the manifest-driven fields.
        layout = DRIAMSLayout(tmp_path, year="2024")
        assert layout.metadata_suffix == "_strat.csv"
        assert layout.mz_min == 2500.0
        assert layout.mz_max == 15000.0
        # User did not set id_column either -> library default (None = auto-detect).
        assert layout.id_column == "code"

    def test_missing_manifest_falls_back_to_library_defaults(self, tmp_path: Path):
        # Dataset on disk has the conventional "_clean.csv" suffix.
        _make_fake_dataset(tmp_path, metadata_suffix="_clean.csv")
        # No site_info.json written.

        layout = DRIAMSLayout(tmp_path, year="2024")
        # Library defaults applied (the previous behaviour, fully backward-compatible).
        assert layout.metadata_suffix == "_clean.csv"
        assert layout.metadata_dir == "id"
        assert layout.spectrum_ext == ".txt"
        assert layout.mz_min == 2000.0
        assert layout.mz_max == 19997.0


# ------------------------------------------------ DatasetBuilder writes manifest


class TestDatasetBuilderWritesManifest:
    """Verify that an end-to-end build emits a well-formed manifest.

    Uses :class:`FlatLayout` over a tiny synthetic dataset.
    """

    def _write_fake_spectrum(self, path: Path) -> None:
        """Write a two-column m/z / intensity text spectrum."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["2000 0.1", "5000 0.5", "10000 0.9", "15000 0.2", "19000 0.05"]
        path.write_text("\n".join(lines))

    def test_build_emits_manifest(self, tmp_path: Path):
        from maldiamrkit.data import DatasetBuilder, FlatLayout

        spectra_dir = tmp_path / "raw_spectra"
        out_dir = tmp_path / "built"
        out_dir.mkdir()

        # Two fake spectra
        self._write_fake_spectrum(spectra_dir / "a.txt")
        self._write_fake_spectrum(spectra_dir / "b.txt")

        # Minimal metadata
        meta_csv = tmp_path / "meta.csv"
        meta_csv.write_text("ID,Species\na,Escherichia coli\nb,Escherichia coli\n")

        layout = FlatLayout(spectra_dir, meta_csv)
        builder = DatasetBuilder(layout, out_dir, on_error="warn")
        report = builder.build()

        assert report.total == 2

        # Manifest written, well-formed, and round-trippable.
        info = read_site_info(out_dir)
        assert info is not None
        assert info.format_version == CURRENT_FORMAT_VERSION
        assert info.metadata_suffix == "_clean.csv"
        assert "raw" in info.spectra_folders
        assert any(f.startswith("binned_") for f in info.spectra_folders)
        # build_info is populated.
        assert info.build_info is not None
        assert info.build_info.source_layout == "FlatLayout"
        assert info.build_info.n_total_spectra == 2

    def test_layout_loads_after_build_without_extra_kwargs(self, tmp_path: Path):
        """End-to-end: build a dataset, then DRIAMSLayout opens it with no kwargs."""
        from maldiamrkit.data import DatasetBuilder, FlatLayout

        spectra_dir = tmp_path / "raw_spectra"
        out_dir = tmp_path / "built"
        out_dir.mkdir()
        self._write_fake_spectrum(spectra_dir / "a.txt")
        self._write_fake_spectrum(spectra_dir / "b.txt")
        meta_csv = tmp_path / "meta.csv"
        meta_csv.write_text("ID,Species\na,Escherichia coli\nb,Escherichia coli\n")

        DatasetBuilder(
            FlatLayout(spectra_dir, meta_csv), out_dir, on_error="warn"
        ).build()

        layout = DRIAMSLayout(out_dir)
        # The manifest gave the loader all it needs for these fields.
        assert layout.metadata_suffix == "_clean.csv"
        assert layout.spectrum_ext == ".txt"
        assert layout.metadata_dir == "id"
