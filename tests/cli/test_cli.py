"""Unit tests for the CLI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from maldiamrkit.cli import app
from maldiamrkit.preprocessing import PreprocessingPipeline

DATA_DIR = Path(__file__).parent.parent.parent / "data"
runner = CliRunner()


class TestHelpOutput:
    """Tests for CLI help and subcommand discovery."""

    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "preprocess" in result.output
        assert "quality" in result.output

    def test_preprocess_help(self):
        result = runner.invoke(app, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "--input-dir" in result.output
        assert "--output" in result.output
        assert "--bin-width" in result.output
        assert "--method" in result.output

    def test_quality_help(self):
        result = runner.invoke(app, ["quality", "--help"])
        assert result.exit_code == 0
        assert "--input-dir" in result.output
        assert "--output" in result.output


class TestPreprocessCommand:
    """Tests for the preprocess subcommand."""

    def test_preprocess_runs(self, tmp_path: Path):
        if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.txt")):
            pytest.skip("No spectrum files in data directory")

        output = tmp_path / "test_preprocess_output.csv"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "--input-dir",
                str(DATA_DIR),
                "--output",
                str(output),
                "--bin-width",
                "3",
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
        df = pd.read_csv(output, index_col=0)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        # Column names should be numeric m/z bin centers
        for col in df.columns:
            float(col)  # raises ValueError if not numeric
        # Index should be named "ID"
        assert df.index.name == "ID"
        # All values should be numeric, non-NaN, and non-negative
        assert df.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()
        assert not df.isna().any().any()
        assert (df.values >= 0).all()

    def test_preprocess_empty_dir(self, tmp_path: Path):
        output = tmp_path / "test_empty.csv"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "--input-dir",
                str(tmp_path),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 1

    def test_preprocess_with_save_spectra_dir(self, tmp_path: Path):
        if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.txt")):
            pytest.skip("No spectrum files in data directory")

        output = tmp_path / "matrix.csv"
        spectra_dir = tmp_path / "preprocessed"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "--input-dir",
                str(DATA_DIR),
                "--output",
                str(output),
                "--save-spectra-dir",
                str(spectra_dir),
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
        assert spectra_dir.exists()
        txt_files = list(spectra_dir.glob("*.txt"))
        assert len(txt_files) > 0
        # Verify tab-separated content
        df = pd.read_csv(txt_files[0], sep="\t")
        assert "mass" in df.columns
        assert "intensity" in df.columns

    def test_preprocess_with_pipeline_json(self, tmp_path: Path):
        if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.txt")):
            pytest.skip("No spectrum files in data directory")

        pipe_path = tmp_path / "test_pipe.json"
        PreprocessingPipeline.default().to_json(pipe_path)

        output = tmp_path / "test_preprocess_pipeline.csv"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "--input-dir",
                str(DATA_DIR),
                "--output",
                str(output),
                "--pipeline",
                str(pipe_path),
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
        df = pd.read_csv(output, index_col=0)
        assert df.shape[0] > 0


class TestQualityCommand:
    """Tests for the quality subcommand."""

    def test_quality_runs(self, tmp_path: Path):
        if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.txt")):
            pytest.skip("No spectrum files in data directory")

        output = tmp_path / "test_quality_report.csv"
        result = runner.invoke(
            app,
            [
                "quality",
                "--input-dir",
                str(DATA_DIR),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
        df = pd.read_csv(output)
        # All expected columns present
        expected_cols = [
            "ID",
            "snr",
            "total_ion_count",
            "peak_count",
            "baseline_fraction",
            "noise_level",
            "dynamic_range",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        assert df.shape[0] > 0
        # No NaN in any column
        assert not df.isna().any().any()
        # Numeric columns should be non-negative and not NaN
        numeric_cols = [c for c in expected_cols if c != "ID"]
        for col in numeric_cols:
            assert (df[col] >= 0).all()
        # SNR may be inf (empty noise region), but other columns should be finite
        finite_cols = [c for c in numeric_cols if c != "snr"]
        for col in finite_cols:
            assert np.all(np.isfinite(df[col].values))
        # baseline_fraction should be in [0, 1]
        assert (df["baseline_fraction"] <= 1).all()
        # SNR should be positive (inf is positive)
        assert (df["snr"] > 0).all()

    def test_quality_empty_dir(self, tmp_path: Path):
        output = tmp_path / "test_quality_empty.csv"
        result = runner.invoke(
            app,
            [
                "quality",
                "--input-dir",
                str(tmp_path),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 1
